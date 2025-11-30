#include "kv_core.hpp"
#include <ATen/ATen.h>

KVEmbeddingCore::KVEmbeddingCore(int dim, double mean, double std) 
    : embedding_dim_(dim), init_mean_(mean), init_std_(std) {}

torch::Tensor KVEmbeddingCore::fetch_vector(int64_t key) {
    auto it = embeddings_.find(key);
    if (it != embeddings_.end()) {
        return it->second;
    } else {
        auto new_vector = torch::empty({embedding_dim_});
        new_vector.normal_(init_mean_, init_std_);
        embeddings_[key] = new_vector;
        return new_vector;
    }
}

void KVEmbeddingCore::accumulate_gradients(
    const std::vector<int64_t>& unique_indices,
    const torch::Tensor& flat_inverse_indices,
    const torch::Tensor& flat_grad_output) {
    
    for (size_t i = 0; i < unique_indices.size(); ++i) {
        int64_t key = unique_indices[i];
        
        // Create mask for this key
        auto mask = (flat_inverse_indices == static_cast<int64_t>(i));
        if (torch::any(mask).item<bool>()) {
            auto selected_grads = flat_grad_output.index({mask});
            auto grad_for_key = selected_grads.mean(0);
            grad_accumulator_[key] = grad_for_key;
        }
    }
}

void KVEmbeddingCore::apply_adam_updates(
    double lr, double beta1, double beta2, 
    double epsilon, double weight_decay) {
    
    for (auto& pair : grad_accumulator_) {
        int64_t key = pair.first;
        torch::Tensor grad = pair.second;
        
        // Apply weight decay
        if (weight_decay > 0) {
            grad = grad + weight_decay * embeddings_[key];
        }
        
        // Initialize momentum states if needed
        if (momentum_states_.find(key) == momentum_states_.end()) {
            momentum_states_[key] = std::make_tuple(
                torch::zeros_like(grad),
                torch::zeros_like(grad),
                0
            );
        }
        
        auto& momentum_state = momentum_states_[key];
        auto& first_momentum = std::get<0>(momentum_state);
        auto& second_momentum = std::get<1>(momentum_state);
        auto& time_step = std::get<2>(momentum_state);
        
        // Update time step
        time_step += 1;
        
        // Update biased first moment estimate
        first_momentum = beta1 * first_momentum + (1 - beta1) * grad;
        
        // Update biased second raw moment estimate
        second_momentum = beta2 * second_momentum + (1 - beta2) * (grad * grad);
        
        // Compute bias-corrected first moment estimate
        auto m_hat = first_momentum / (1 - std::pow(beta1, time_step));
        
        // Compute bias-corrected second raw moment estimate
        auto v_hat = second_momentum / (1 - std::pow(beta2, time_step));
        
        // Update parameter
        embeddings_[key] -= lr * m_hat / (torch::sqrt(v_hat) + epsilon);
    }
    
    clear_gradients();
}

void KVEmbeddingCore::clear_gradients() {
    grad_accumulator_.clear();
}

std::vector<int64_t> KVEmbeddingCore::get_keys() {
    std::vector<int64_t> keys;
    keys.reserve(embeddings_.size());
    for (const auto& pair : embeddings_) {
        keys.push_back(pair.first);
    }
    return keys;
}

int64_t KVEmbeddingCore::get_embedding_count() {
    return embeddings_.size();
}