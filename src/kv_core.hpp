#pragma once

// #include <unordered_map>
#include <vector>
#include <tuple>
#include <torch/torch.h>
#include "absl/container/flat_hash_map.h"

class KVEmbeddingCore {
private:
    // std::unordered_map<int64_t, torch::Tensor> embeddings_;
    // std::unordered_map<int64_t, std::tuple<torch::Tensor, torch::Tensor, int>> momentum_states_;
    // std::unordered_map<int64_t, torch::Tensor> grad_accumulator_;
    // 使用 absl::flat_hash_map 替代 std::unordered_map
    absl::flat_hash_map<int64_t, torch::Tensor> embeddings_;
    absl::flat_hash_map<int64_t, std::tuple<torch::Tensor, torch::Tensor, int>> momentum_states_;
    absl::flat_hash_map<int64_t, torch::Tensor> grad_accumulator_;
    
    
    int embedding_dim_;
    double init_mean_, init_std_;

public:
    KVEmbeddingCore(int dim, double mean, double std);
    
    torch::Tensor fetch_vector(int64_t key);
    void accumulate_gradients(
        const std::vector<int64_t>& unique_indices,
        const torch::Tensor& flat_inverse_indices,
        const torch::Tensor& flat_grad_output
    );
    
    // Optimizer functions
    void apply_adam_updates(
        double lr, double beta1, double beta2, 
        double epsilon, double weight_decay
    );
    
    void clear_gradients();
    std::vector<int64_t> get_keys();
    int64_t get_embedding_count();
    
    // Accessor methods for debugging
    // std::unordered_map<int64_t, torch::Tensor>& get_embeddings() { return embeddings_; }
    // std::unordered_map<int64_t, torch::Tensor>& get_grad_accumulator() { return grad_accumulator_; }
    absl::flat_hash_map<int64_t, torch::Tensor>& get_embeddings() { return embeddings_; }
    absl::flat_hash_map<int64_t, torch::Tensor>& get_grad_accumulator() { return grad_accumulator_; }

};