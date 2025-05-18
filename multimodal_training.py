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


class MultimodalDataset(Dataset):
    def __init__(self, pet_dir, ct_dir, csv_path, transform=None):
        self.pet_dir = pet_dir
        self.ct_dir = ct_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform

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
        pet_data = self.load_slices(pet_dir_path)
        ct_data = self.load_slices(ct_dir_path)

        # Extract clinical text features
        text_features = row.drop(['Patient #']).values.astype(np.float32)

        if self.transform:
            pet_data = self.transform(pet_data)
            ct_data = self.transform(ct_data)

        return (
            torch.tensor(pet_data, dtype=torch.float32), 
            torch.tensor(ct_data, dtype=torch.float32), 
            torch.tensor(text_features, dtype=torch.float32)
        )


class MultimodalModel(nn.Module):
    def __init__(self, mae_model, fusion_model, text_embedder, qformer):
        super(MultimodalModel, self).__init__()
        self.mae = mae_model
        self.fusion = fusion_model
        self.text_embedder = text_embedder
        self.qformer = qformer

    def forward(self, pet, ct, text):
        print("PET input shape:", pet.shape)
        print("CT input shape:", ct.shape)
        pet_features = self.mae.extract_features(pet)
        ct_features = self.mae.extract_features(ct)

        fused_features = self.fusion(pet_features, ct_features)
        embedded_text = self.text_embedder.get_embedded_text(text)

        multimodal_features = self.qformer(fused_features, embedded_text)

        return multimodal_features

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, device, warmup_epochs=10, total_epochs=300):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0

        for pet, ct, text in self.dataloader:
            
            pet, ct, text = pet.to(self.device), ct.to(self.device), text.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(pet, ct, text)

            # Assuming text contains time and event information
            time = text[:, -2]
            event = text[:, -1]
            loss = self.criterion(outputs.squeeze(), time, event)
            loss.backward()
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
                outputs = self.model(pet, ct, text).squeeze().cpu().numpy()

                times = text[:, -2].cpu().numpy()
                events = text[:, -1].cpu().numpy()

                all_preds.extend(outputs)
                all_times.extend(times)
                all_events.extend(events)

        all_preds = np.array(all_preds)
        all_times = np.array(all_times)
        all_events = np.array(all_events)

        # Using negative predictions since higher values indicate higher risk
        c_index = concordance_index(all_times, -all_preds, all_events)
        return c_index

    def train(self):
        for epoch in range(self.total_epochs):
            epoch_loss = self.train_one_epoch(epoch)
            print(f"Epoch {epoch+1}/{self.total_epochs}, Loss: {epoch_loss}")

            # Calculate C-index every epoch
            c_index = self.evaluate_c_index()
            print(f"Epoch {epoch+1}/{self.total_epochs}, C-index: {c_index}")

if __name__ == "__main__":
    # Data paths
    pet_dir = "/Users/varun.t1/Documents/vat_base/hnc_project/numpy_files/train/PET"
    ct_dir = "/Users/varun.t1/Documents/vat_base/hnc_project/numpy_files/train/CT"
    csv_path = "/Users/varun.t1/Documents/vat_base/hnc_project/numpy_files/train/csv/hnc_train_sample_preprocessed.csv"

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
    input_dim = 33  # Assuming 33 clinical features
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

    # Downstream Phase
    downstream_dataloader = DataLoader(pretrain_dataset, batch_size=256, shuffle=True)
    downstream_model = MultimodalModel(mae_model, fusion_model, text_embedder, qformer)
    downstream_model.to(device)

    downstream_optimizer = optim.AdamW(downstream_model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-2)
    downstream_trainer = Trainer(downstream_model, downstream_dataloader, downstream_optimizer, criterion, device)
    downstream_trainer.train()
