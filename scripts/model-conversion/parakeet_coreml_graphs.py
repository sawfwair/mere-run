"""Equivalent Parakeet graphs expressed with ANE-supported operations."""

import torch


class ParakeetANEEncoder(torch.nn.Module):
    """Preserve encoder math while keeping mask construction in FP16."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor):
        # All mask lengths are exact integers below 2,048 in this fixed geometry.
        hidden = input_features.unsqueeze(1)
        lengths = attention_mask.sum(-1)
        for layer in self.encoder.subsampling.layers:
            hidden = layer(hidden)
            if isinstance(layer, torch.nn.Conv2d):
                lengths = torch.floor(
                    (lengths + layer.padding[0] * 2 - layer.kernel_size[0])
                    / layer.stride[0]
                ) + 1
                indices = torch.arange(hidden.shape[2], dtype=hidden.dtype)
                valid = indices[None, :] < lengths[:, None]
                hidden = torch.where(
                    valid[:, None, :, None], hidden, torch.zeros_like(hidden)
                )
        hidden = hidden.transpose(1, 2).reshape(
            hidden.shape[0], hidden.shape[2], -1
        )
        hidden = self.encoder.subsampling.linear(hidden) * self.encoder.input_scale
        positions = self.encoder.encode_positions(hidden)
        indices = torch.arange(hidden.shape[1], dtype=hidden.dtype)
        valid = indices[None, :] < lengths[:, None]
        # max(i, j) < length is equivalent to valid(i) AND valid(j). The
        # floating-point comparison avoids the CPU-only boolean AND operation.
        pair_indices = torch.maximum(indices[:, None], indices[None, :])
        pair_mask = pair_indices[None, None, :, :] < lengths[:, None, None, None]
        for layer in self.encoder.layers:
            hidden = hidden + 0.5 * layer.feed_forward1(
                layer.norm_feed_forward1(hidden)
            )
            attention, _ = layer.self_attn(
                hidden_states=layer.norm_self_att(hidden),
                attention_mask=pair_mask,
                position_embeddings=positions,
            )
            hidden = hidden + attention
            conv = layer.conv
            value = conv.pointwise_conv1(layer.norm_conv(hidden).transpose(1, 2))
            value = torch.nn.functional.glu(value, dim=1)
            # The padding mask is an outer product. Its row validity is already
            # known, so no boolean reduction is needed inside each convolution.
            value = torch.where(valid[:, None, :], value, torch.zeros_like(value))
            value = conv.pointwise_conv2(
                conv.activation(conv.norm(conv.depthwise_conv(value)))
            )
            hidden = hidden + value.transpose(1, 2)
            hidden = hidden + 0.5 * layer.feed_forward2(
                layer.norm_feed_forward2(hidden)
            )
            hidden = layer.norm_out(hidden)
        output_mask = torch.where(
            valid,
            torch.ones_like(valid, dtype=hidden.dtype),
            torch.zeros_like(valid, dtype=hidden.dtype),
        )
        return hidden, output_mask


def first_small_index(scores: torch.Tensor) -> torch.Tensor:
    """Return the first maximum's index for at most 128 finite scores."""
    indices = torch.arange(scores.shape[-1]).to(scores.dtype).expand_as(scores)
    winners = scores == torch.amax(scores, dim=-1, keepdim=True)
    return torch.amin(
        torch.where(winners, indices, torch.full_like(scores, 128.0)),
        dim=-1,
        keepdim=True,
    )


def first_index_digits(scores: torch.Tensor) -> torch.Tensor:
    """Encode the first maximum as two exact FP16 base-128 digits.

    FP16 cannot represent every vocabulary index directly. Selecting the first
    group, then the first index within that group, preserves argmax's tie rule
    without an integer reduction or an inexact cast of the final token ID.
    """
    count = scores.shape[-1]
    if count <= 128:
        low = first_small_index(scores)
        return torch.cat([torch.zeros_like(low), low], dim=-1)
    groups = (count + 127) // 128
    padding = torch.full(
        (*scores.shape[:-1], groups * 128 - count),
        float("-inf"),
        dtype=scores.dtype,
    )
    # Concatenation maps to the ANE; the general pad operator does not.
    padded = torch.cat([scores, padding], dim=-1)
    grouped = padded.reshape(*scores.shape[:-1], groups, 128)
    high = first_small_index(torch.amax(grouped, dim=-1))
    group_indices = torch.arange(groups).to(scores.dtype).reshape(1, 1, groups, 1)
    selected = torch.where(
        group_indices == high.unsqueeze(-1), grouped, torch.zeros_like(grouped)
    )
    low = first_small_index(torch.sum(selected, dim=-2))
    return torch.cat([high, low], dim=-1)
