import torch


def build_2d_sin_cos_embed(
    height,
    width,
    embed_dim,
    temperature=10000.0,
    device=None,
    dtype=torch.float32,
):
    assert embed_dim % 4 == 0

    num_freqs = embed_dim // 4

    freq_indices = torch.arange(
        num_freqs,
        device=device,
        dtype=dtype,
    )

    inv_freq = temperature ** (
        -freq_indices / num_freqs
    )

    x = torch.arange(
        width,
        device=device,
        dtype=dtype,
    )

    y = torch.arange(
        height,
        device=device,
        dtype=dtype,
    )

    x_embed = inv_freq[:, None, None] * x[None, None, :]
    y_embed = inv_freq[:, None, None] * y[None, :, None]

    sin_x = x_embed.sin().expand(-1, height, -1)
    cos_x = x_embed.cos().expand(-1, height, -1)
    sin_y = y_embed.sin().expand(-1, -1, width)
    cos_y = y_embed.cos().expand(-1, -1, width)

    pos_embed = torch.cat(
        [sin_x, cos_x, sin_y, cos_y],
        dim=0,
    )

    return pos_embed.flatten(1).transpose(0, 1)
