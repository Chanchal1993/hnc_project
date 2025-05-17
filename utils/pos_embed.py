import numpy as np

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim (int): The embedding dimension for each position.
        grid_size (int): The height and width of the grid (assumed square).
        cls_token (bool): If True, prepend a zero vector for the class token.

    Returns:
        pos_embed: numpy array of shape (grid_size*grid_size+1, embed_dim) if cls_token else (grid_size*grid_size, embed_dim)
    """
    # Helper function to generate 1D sine-cosine embeddings
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / (10000 ** omega)  # (D/2,)

        pos = pos.reshape(-1)  # flatten if needed
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2)
        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    # Create grid of positions
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # two 2D arrays (H, W)

    grid = np.stack(grid, axis=0)  # (2, H, W)
    grid = grid.reshape(2, -1)  # (2, H*W)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

    if cls_token:
        cls_token_embed = np.zeros([1, embed_dim], dtype=np.float32)
        pos_embed = np.vstack([cls_token_embed, pos_embed])

    return pos_embed
