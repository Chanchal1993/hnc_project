import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from lifelines.utils import concordance_index
from cox_survival_loss import survival_cox_sigmoid_loss
from fusion_module import ClinicalTextEmbedder, CrossAttentionFusion, QFormer
from masked_autoencoder_vit import MaskedAutoencoderViT, mae_vit_base_patch16
from torchvision import transforms


class MultimodalDataset(Dataset):
    def __init__(self, pet_dir, ct_dir, csv_path, transform=None):
        self.pet_dir = pet_dir
        self.ct_dir = ct_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.resize = transforms.Resize((128, 128))

    def __len__(self):
        return len(self.data)

    def load_slices(self, dir_path):
        """ Load all slices for a patient and stack them along the depth axis. """
        slice_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy')])
        slices = [np.load(os.path.join(dir_path, f)) for f in slice_files]
        volume = np.stack(slices, axis=0)  # Stack along depth axis
        return volume

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = row['Patient #']

        pet_dir_path = os.path.join(self.pet_dir, patient_id)
        ct_dir_path = os.path.join(self.ct_dir, patient_id)

        # Load slices and create 3D volume
        pet_data = self.load_slices(pet_dir_path)  # [slices, height, width, 1]
        ct_data = self.load_slices(ct_dir_path)    # [slices, height, width, 1]

        # Use the middle slice for 2D input
        mid_idx = pet_data.shape[0] // 2
        pet_data = pet_data[mid_idx]  # shape: [height, width, 1]
        ct_data = ct_data[mid_idx]    # shape: [height, width, 1]

        # Move channel to second position for PyTorch
        pet_data = np.transpose(pet_data, (2, 0, 1))  # [1, height, width]
        ct_data = np.transpose(ct_data, (2, 0, 1))    # [1, height, width]

        # Convert to torch tensors for resizing
        pet_data = torch.from_numpy(pet_data).float().requires_grad_(True)
        ct_data = torch.from_numpy(ct_data).float().requires_grad_(True)

        # Resize to 128x128
        pet_data = self.resize(pet_data)  # [1, 128, 128]
        ct_data = self.resize(ct_data)    # [1, 128, 128]

        # Extract clinical text features
        text_features = row.drop(['Patient #']).values.astype(np.float32)
        # Check for NaNs and replace with zero
        if np.isnan(text_features).any():
            print(f"Warning: NaNs found in clinical features for patient {patient_id}. Replacing with zeros.")
            text_features = np.nan_to_num(text_features, nan=0.0)

        if self.transform:
            pet_data = self.transform(pet_data)
            ct_data = self.transform(ct_data)

        return (
            pet_data,  # [1, 128, 128]
            ct_data,   # [1, 128, 128]
            torch.tensor(text_features, dtype=torch.float32, requires_grad=True)
        )


class MultimodalModel(nn.Module):
    def __init__(self, mae_model, fusion_model, text_embedder, qformer):
        super(MultimodalModel, self).__init__()
        self.mae = mae_model
        # Ensure MAE parameters require gradients
        for param in self.mae.parameters():
            param.requires_grad = True
            
        self.fusion = fusion_model
        self.text_embedder = text_embedder
        self.qformer = qformer
        # Add a final prediction layer
        self.prediction_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, pet, ct, text):
        # Ensure inputs require gradients
        pet = pet.detach().requires_grad_(True)
        ct = ct.detach().requires_grad_(True)
        text = text.detach().requires_grad_(True)
        
        print("PET input shape:", pet.shape)
        print("CT input shape:", ct.shape)
        
        pet_features = self.mae.extract_features(pet)
        ct_features = self.mae.extract_features(ct)

        fused_features = self.fusion(pet_features, ct_features)
        embedded_text = self.text_embedder.get_embedded_text(text)

        print('Model: fused_features shape:', fused_features.shape)
        print('Model: embedded_text shape:', embedded_text.shape)

        multimodal_features = self.qformer(fused_features, embedded_text)
        
        # Take mean across sequence dimension and apply prediction head
        pooled_features = multimodal_features.mean(dim=0)  # (batch_size, embed_dim)
        predictions = self.prediction_head(pooled_features)  # (batch_size, 1)
        
        return predictions.squeeze(-1)  # (batch_size,)

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, device, warmup_epochs=10, total_epochs=30):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        
        # Ensure all model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0

        for pet, ct, text in self.dataloader:
            # Move to device and ensure gradients
            pet = pet.to(self.device).requires_grad_(True)
            ct = ct.to(self.device).requires_grad_(True)
            text = text.to(self.device).requires_grad_(True)
            
            self.optimizer.zero_grad()
            
            # Get predictions
            predictions = self.model(pet, ct, text)
            
            # Extract time and event information
            time = text[:, -2]  # Time to event
            event = text[:, -1]  # Event indicator
            
            # Calculate loss
            loss = self.criterion(predictions, time, event)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected. Skipping batch.")
                continue
            
            # Ensure loss requires grad
            if not loss.requires_grad:
                print("Warning: Loss doesn't require grad. Adding requires_grad=True")
                loss.requires_grad_(True)
                
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            epoch_loss += loss.item()

        # Warm-up learning rate adjustment
        if epoch < self.warmup_epochs:
            self.scheduler.step()

        return epoch_loss / len(self.dataloader)

    def evaluate_c_index(self):
        """ Calculate C-index using the entire dataset """
        self.model.eval()
        all_times, all_events, all_preds = [], [], []

        with torch.no_grad():
            for pet, ct, text in self.dataloader:
                pet, ct, text = pet.to(self.device), ct.to(self.device), text.to(self.device)
                predictions = self.model(pet, ct, text)
                
                times = text[:, -2].cpu().numpy()
                events = text[:, -1].cpu().numpy()
                preds = predictions.cpu().numpy()

                all_preds.extend(preds)
                all_times.extend(times)
                all_events.extend(events)

        all_preds = np.array(all_preds)
        all_times = np.array(all_times)
        all_events = np.array(all_events)

        # Print data distribution for debugging
        print(f"\nData Distribution:")
        print(f"Number of samples: {len(all_times)}")
        print(f"Number of events: {np.sum(all_events)}")
        print(f"Unique times: {np.unique(all_times)}")
        print(f"Unique events: {np.unique(all_events)}")

        # Ensure all arrays have the same shape
        assert len(all_preds) == len(all_times) == len(all_events), \
            f"Shape mismatch: preds={len(all_preds)}, times={len(all_times)}, events={len(all_events)}"

        try:
            # Using negative predictions since higher values indicate higher risk
            c_index = concordance_index(all_times, -all_preds, all_events)
            return c_index
        except ZeroDivisionError as e:
            print("\nWarning: Could not calculate C-index due to insufficient data variation.")
            print("This usually happens when:")
            print("1. All events are the same (all 0 or all 1)")
            print("2. All times are the same")
            print("3. The dataset is too small to form valid pairs")
            return 0.5  # Return random performance as baseline

    def train(self):
        for epoch in range(self.total_epochs):
            epoch_loss = self.train_one_epoch(epoch)
            print(f"Epoch {epoch+1}/{self.total_epochs}, Loss: {epoch_loss}")

            # Calculate C-index every epoch
            c_index = self.evaluate_c_index()
            print(f"Epoch {epoch+1}/{self.total_epochs}, C-index: {c_index}")

if __name__ == "__main__":
    # Data paths
    pet_dir = "//Users/chanchalm/HNCmodel_new/numpy_files/train/PET"
    ct_dir = "/Users/chanchalm/HNCmodel_new/numpy_files/train/CT"
    csv_path = "/Users/chanchalm/HNCmodel_new/numpy_files/train/csv/hnc_train_sample_preprocessed.csv"

    # Pretraining Phase
    pretrain_dataset = MultimodalDataset(pet_dir, ct_dir, csv_path)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=128, shuffle=True)

    # Initialize model components (implement/load actual models)
    # MAE Model with specific parameters
    mae_model = mae_vit_base_patch16(img_size=128, in_chans=1)



# Fusion Model
    embed_dim = 768
    num_heads = 8
    fusion_model = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads)

# Text Embedder
    input_dim = 22  # Updated to match the actual number of clinical features in the CSV
    text_embedder = ClinicalTextEmbedder(input_dim=input_dim, embed_dim=embed_dim)

# QFormer
    num_queries = 32
    qformer = QFormer(embed_dim=embed_dim, num_queries=num_queries, num_heads=num_heads)

    pretrain_model = MultimodalModel(mae_model, fusion_model, text_embedder, qformer)
    optimizer = optim.AdamW(pretrain_model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-2)
    criterion = survival_cox_sigmoid_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_model.to(device)

    # Pretraining
    pretrainer = Trainer(pretrain_model, pretrain_dataloader, optimizer, criterion, device, warmup_epochs=0, total_epochs=1)
    pretrainer.train()

     # Save the pretrained model
    # Save the pretrained model
    torch.save(pretrain_model.state_dict(), "pretrained_model.pth")
    print("Pretrained model saved as pretrained_model.pth")


    # Downstream Phase
    downstream_dataloader = DataLoader(pretrain_dataset, batch_size=256, shuffle=True)
    downstream_model = MultimodalModel(mae_model, fusion_model, text_embedder, qformer)
    downstream_model.to(device)

    downstream_optimizer = optim.AdamW(downstream_model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-2)
    downstream_trainer = Trainer(downstream_model, downstream_dataloader, downstream_optimizer, criterion, device)
    downstream_trainer.train()

    # Save the downstream model
    # Save the downstream model
    torch.save(downstream_model.state_dict(), "downstream_model.pth")
    print("Downstream model saved as downstream_model.pth")

