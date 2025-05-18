import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from cox_survival_loss import survival_cox_sigmoid_loss
from fusion_module import ClinicalTextEmbedder, CrossAttentionFusion, QFormer
from masked_autoencoder_vit import MaskedAutoencoderViT, mae_vit_base_patch16
from torchvision import transforms
from multimodal_training import MultimodalModel, MultimodalDataset


class Tester:
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0.0
        all_times, all_events, all_preds = [], [], []

        with torch.no_grad():
            for pet, ct, text in self.dataloader:
                pet, ct, text = pet.to(self.device), ct.to(self.device), text.to(self.device)
                predictions = self.model(pet, ct, text)

                # Extract time and event information
                times = text[:, -2].cpu().numpy()
                events = text[:, -1].cpu().numpy()
                preds = predictions.cpu().numpy()

                all_preds.extend(preds)
                all_times.extend(times)
                all_events.extend(events)

                # Calculate loss
                loss = self.criterion(predictions, text[:, -2], text[:, -1])
                epoch_loss += loss.item()

        # Calculate average loss
        avg_loss = epoch_loss / len(self.dataloader)

        # Calculate C-index
        try:
            c_index = concordance_index(all_times, -np.array(all_preds), all_events)
        except ZeroDivisionError:
            print("Warning: Unable to calculate C-index due to lack of variability in the data.")
            c_index = 0.5

        print(f"Test Loss: {avg_loss:.4f}, C-index: {c_index:.4f}")
        return avg_loss, c_index


if __name__ == "__main__":
    # Paths to test data
    pet_dir = "/Users/chanchalm/HNCmodel_new/numpy_files/test/PET"
    ct_dir = "/Users/chanchalm/HNCmodel_new/numpy_files/test/CT"
    csv_path = "/Users/chanchalm/HNCmodel_new/numpy_files/test/csv/hnc_test_sample_preprocessed.csv"

    # Initialize dataset and dataloader
    test_dataset = MultimodalDataset(pet_dir, ct_dir, csv_path)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Model components
    embed_dim = 768
    num_heads = 8
    input_dim = 22
    num_queries = 32

    # Load model components
    mae_model = mae_vit_base_patch16(img_size=128, in_chans=1)
    fusion_model = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads)
    text_embedder = ClinicalTextEmbedder(input_dim=input_dim, embed_dim=embed_dim)
    qformer = QFormer(embed_dim=embed_dim, num_queries=num_queries, num_heads=num_heads)

    # Initialize model
    model = MultimodalModel(mae_model, fusion_model, text_embedder, qformer)
    model.load_state_dict(torch.load("/Users/chanchalm/HNCmodel_new/downstream_model.pth"))  # Update with actual path
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Initialize tester
    criterion = survival_cox_sigmoid_loss
    tester = Tester(model, test_dataloader, criterion, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Run evaluation
    tester.evaluate()

