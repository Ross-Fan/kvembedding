import torch
from typing import List
from .kv_embedding import KVEmbedding


class KVAdamOptimizer:
    def __init__(self, kv_embedding: KVEmbedding, **kwargs):
        self.kv_embedding = kv_embedding
        self.learning_rate = kwargs.get('lr', 0.001)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.weight_decay = kwargs.get('weight_decay', 0.0)
    
    def step(self):
        """Perform one optimization step using C++ backend."""
        self.kv_embedding.cpp_backend.apply_adam_updates(
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.weight_decay
        )
    
    def zero_grad(self):
        """Clear gradients."""
        self.kv_embedding.cpp_backend.clear_gradients()