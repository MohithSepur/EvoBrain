"""
Adapted from https://github.com/hugochan/IDGL/blob/master/src/core/layers/graphlearn.py
Author: hugochan
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VERY_SMALL_NUMBER = 1e-12
INF = 1e20



class GraphLearner(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_nodes,
        num_heads=1,
        embed_dim=10,
        metric_type="self_attention",
    ):
        super(GraphLearner, self).__init__()

        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.metric_type = metric_type

        if metric_type == "weighted_cosine":
            self.weight_tensor = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(num_heads, input_size))
            )

        elif metric_type == "self_attention":
            self.linear_Q = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_K = nn.Linear(input_size, hidden_size, bias=False)

        elif metric_type == "cosine": # no learnable params
            pass

        elif metric_type == "adaptive":
            # for adaptive GSL, each "head" is for one graph within a temporal resolution
            if num_heads > 1:
                self.E1 = nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(num_heads, num_nodes, embed_dim))
                )
            else:
                self.E1 = nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(num_nodes, embed_dim))
                )

        else:
            raise ValueError("Unknown metric_type: {}".format(metric_type))

    @staticmethod
    def _apply_mask(attention, attn_mask, markoff_value):
        """Apply attention mask with automatic dimension alignment."""
        if attn_mask is not None:
            while attn_mask.dim() < attention.dim():
                attn_mask = attn_mask.unsqueeze(0)
            attention = attention.masked_fill(
                torch.logical_not(attn_mask.bool()), markoff_value
            )
        return attention

    def forward(self, context, attn_mask=None, batch_size=None):
        """
        Args:
            context: (batch, num_nodes, dim) or (num_nodes, dim)
            attn_mask: (batch, num_nodes, num_nodes) or (num_nodes, num_nodes),
                       0 will be masked out
            batch_size: required for 'adaptive' metric_type (context-independent)
        Returns:
            attention: (batch, num_nodes, num_nodes)
                       or (batch, num_heads, num_nodes, num_nodes) for adaptive
                       with num_heads > 1
        """
        if self.metric_type == "weighted_cosine":
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(
                0
            )
            attention = torch.clamp(attention, min=0)

            attention = self._apply_mask(attention, attn_mask, markoff_value=0)

        elif self.metric_type == "self_attention":
            # Handle unbatched (2D) input — bmm requires 3D
            needs_squeeze = False
            if context.dim() == 2:
                context = context.unsqueeze(0)
                needs_squeeze = True

            Q = self.linear_Q(context)
            K = self.linear_K(context)

            attention = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])

            attention = self._apply_mask(attention, attn_mask, markoff_value=-INF)
            attention = torch.softmax(attention, dim=-1)

            if needs_squeeze:
                attention = attention.squeeze(0)

        elif self.metric_type == "cosine":
            context_norm = F.normalize(context, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2))
            attention = torch.clamp(attention, min=0)

            attention = self._apply_mask(attention, attn_mask, markoff_value=0)

        elif self.metric_type == "adaptive":
            assert batch_size is not None, (
                "adaptive metric requires batch_size to be provided"
            )
            attention = F.leaky_relu(torch.matmul(self.E1, self.E1.transpose(-1,-2)))

            if self.num_heads > 1:
                # (num_heads, N, N) -> (batch_size, num_heads, N, N)
                # Each head is a distinct learned graph for a temporal resolution
                attention = self._apply_mask(attention, attn_mask, markoff_value=-INF)
                attention = torch.softmax(attention, dim=-1)
                attention = attention.unsqueeze(0).expand(batch_size, -1, -1, -1)
            else:
                # (N, N) -> (batch_size, N, N)
                attention = self._apply_mask(attention, attn_mask, markoff_value=-INF)
                attention = torch.softmax(attention, dim=-1)
                attention = attention.unsqueeze(0).expand(batch_size, -1, -1)

        else:
            raise NotImplementedError()

        return attention