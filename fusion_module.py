import torch
import torch.nn as nn
import torch.nn.functional as F

class ClinicalTextEmbedder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(ClinicalTextEmbedder, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, embed_dim)
        self.activation = nn.ReLU()

    def get_embedded_text(self, clinical_features):
        # clinical_features: (batch_size, input_dim)
        embedded = self.embedding_layer(clinical_features)  # (batch_size, embed_dim)
        embedded = self.activation(embedded)
        embedded = embedded.unsqueeze(0)  # (1, batch_size, embed_dim)
        return embedded

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Multihead attention module
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, pet_features, ct_features):
        Q = self.query_proj(ct_features)
        K = self.key_proj(ct_features)
        V = self.value_proj(pet_features)

        attn_output, _ = self.multihead_attn(Q, K, V)
        fused = self.out_proj(attn_output)
        return fused

class QFormer(nn.Module):
    def __init__(self, embed_dim, num_queries, num_heads):
        super(QFormer, self).__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Learnable query vectors
        self.query_tokens = nn.Parameter(torch.randn(num_queries, 1, embed_dim))

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

    def forward(self, fused_img_features, text_features):
        batch_size = fused_img_features.size(1)
        queries = self.query_tokens.expand(-1, batch_size, -1)

        multimodal_features = self.transformer_decoder(tgt=queries, memory=text_features)

        return multimodal_features

# Example usage within fusion_module.py
if __name__ == "__main__":
    batch_size = 4
    input_dim = 10
    embed_dim = 768
    num_heads = 8
    num_queries = 32

    # Clinical features
    clinical_features = torch.randn(batch_size, input_dim)
    text_embedder = ClinicalTextEmbedder(input_dim, embed_dim)
    embedded_text = text_embedder.get_embedded_text(clinical_features)

    # Image features (dummy data)
    seq_len_pet = 64
    seq_len_ct = 64
    pet_features = torch.randn(seq_len_pet, batch_size, embed_dim)
    ct_features = torch.randn(seq_len_ct, batch_size, embed_dim)

    # Fusion
    fusion_model = CrossAttentionFusion(embed_dim, num_heads)
    fused_features = fusion_model(pet_features, ct_features)

    # QFormer
    q_former = QFormer(embed_dim, num_queries, num_heads)
    multimodal_features = q_former(fused_features, embedded_text)

    print("Embedded text shape:", embedded_text.shape)
    print("Fused features shape:", fused_features.shape)
    print("Multimodal features shape:", multimodal_features.shape)
