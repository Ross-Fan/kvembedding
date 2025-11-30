import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union
from ._C import KVEmbeddingCore  # C++ extension


class KVEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dummy_input, indices, embedding_layer):
        """
        Forward pass for KV embedding lookup.
        """
        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
        ctx.save_for_backward(indices, unique_indices, inverse_indices)
        ctx.embedding_layer = embedding_layer

        # Use C++ backend for efficient embedding lookup
        unique_embeddings = []
        for idx in unique_indices:
            key = idx.item()
            embedding_vector = embedding_layer.cpp_backend.fetch_vector(key)
            unique_embeddings.append(embedding_vector)

        unique_embeddings = torch.stack(unique_embeddings)
        embeddings = unique_embeddings[inverse_indices] + (dummy_input.sum() * 0.0)

        return embeddings
    
    @staticmethod
    def backward(ctx, grad_output):
        indices, unique_indices, inverse_indices = ctx.saved_tensors
        embedding_layer = ctx.embedding_layer

        embedding_layer._accumulate_gradients(unique_indices, inverse_indices, grad_output)

        return embedding_layer._grad_dummy, None, None


class KVEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, mean: float, std: float, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._grad_dummy = nn.Parameter(torch.zeros(1), requires_grad=True)
        
        # Initialize C++ backend
        self.cpp_backend = KVEmbeddingCore(embedding_dim, mean, std)

    def _accumulate_gradients(self, unique_indices, inverse_indices, grad_output):
        flat_grad_output = grad_output.view(-1, self.embedding_dim)
        flat_inverse_indices = inverse_indices.view(-1)
        
        # Delegate to C++ backend
        self.cpp_backend.accumulate_gradients(
            unique_indices.tolist(),
            flat_inverse_indices,
            flat_grad_output
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return KVEmbeddingFunction.apply(self._grad_dummy, indices, self)
    
    def get_embedding_count(self) -> int:
        """Get the number of stored embeddings."""
        return self.cpp_backend.get_embedding_count()