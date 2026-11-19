"""50 - Decoding: greedy, temperature, top-k and top-p

Implement the core decoding algorithms and filters used in LLM generation.
"""

import numpy as np
from typing import List, Optional

def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute the softmax of logits with temperature."""
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive.")
    
    # Scale logits by temperature first
    scaled_logits = logits / temperature
    
    # Subtract max for numerical stability
    shifted_logits = scaled_logits - np.max(scaled_logits)
    
    # Exponentiate and normalize
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits)

def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Filter probabilities keeping only the top k values."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    vocab_size = len(probs)
    if k >= vocab_size:
        return probs.copy()
    
    # Find the indices that would sort the array in descending order
    sorted_indices = np.argsort(probs)[::-1]
    
    # Create a copy of probs to modify
    filtered_probs = probs.copy()
    
    # Zero out elements that are not in the top k
    cutoff_indices = sorted_indices[k:]
    filtered_probs[cutoff_indices] = 0.0
    
    # Renormalize
    sum_probs = np.sum(filtered_probs)
    if sum_probs > 0:
        filtered_probs = filtered_probs / sum_probs
    else:
        filtered_probs[sorted_indices[0]] = 1.0
        
    return filtered_probs

def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Filter probabilities keeping only the top elements whose cumulative probability is >= p."""
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in the range (0.0, 1.0].")
        
    # Find the indices that would sort the array in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Compute cumulative sum of sorted probabilities
    cum_sum = np.cumsum(sorted_probs)
    
    # Find where the cumulative sum exceeds or equals p
    cutoff_idx = np.where(cum_sum >= p)[0]
    if len(cutoff_idx) > 0:
        cutoff_idx = cutoff_idx[0]
    else:
        cutoff_idx = len(probs) - 1
        
    # Create a copy of probs to modify
    filtered_probs = probs.copy()
    
    # Zero out elements that are not in the top-p set
    cutoff_indices = sorted_indices[cutoff_idx + 1:]
    filtered_probs[cutoff_indices] = 0.0
    
    # Renormalize
    sum_probs = np.sum(filtered_probs)
    if sum_probs > 0:
        filtered_probs = filtered_probs / sum_probs
    else:
        filtered_probs[sorted_indices[0]] = 1.0
        
    return filtered_probs

def apply_repetition_penalty(logits: np.ndarray, generated_tokens: List[int], penalty: float) -> np.ndarray:
    """Apply CTRL-style repetition penalty to logits."""
    if penalty < 1.0:
        raise ValueError("Repetition penalty must be >= 1.0.")
        
    modified_logits = logits.copy()
    unique_tokens = set(generated_tokens)
    
    for token in unique_tokens:
        if 0 <= token < len(logits):
            val = modified_logits[token]
            if val >= 0:
                modified_logits[token] = val / penalty
            else:
                modified_logits[token] = val * penalty
                
    return modified_logits

def sample_next_token(probs: np.ndarray, generator: np.random.Generator) -> int:
    """Sample a token index from the probability distribution using the generator."""
    return int(generator.choice(len(probs), p=probs))

def greedy_decode(transition_matrix: np.ndarray, start_token: int, max_length: int) -> List[int]:
    """Perform greedy decoding over a transition log-probability matrix."""
    if max_length <= 0:
        return []
        
    sequence = [start_token]
    current_token = start_token
    
    for _ in range(max_length - 1):
        next_token = int(np.argmax(transition_matrix[current_token]))
        sequence.append(next_token)
        current_token = next_token
        
    return sequence

def beam_search(transition_matrix: np.ndarray, start_token: int, beam_width: int, max_length: int) -> List[int]:
    """Perform beam search decoding over a transition log-probability matrix."""
    if max_length <= 0:
        return []
    if beam_width <= 0:
        raise ValueError("Beam width must be a positive integer.")
        
    vocab_size = transition_matrix.shape[1]
    
    # Each candidate is a tuple: (sequence, cumulative_log_prob)
    candidates = [([start_token], 0.0)]
    
    for _ in range(max_length - 1):
        all_candidates = []
        for seq, score in candidates:
            last_token = seq[-1]
            for next_token in range(vocab_size):
                new_score = score + transition_matrix[last_token, next_token]
                all_candidates.append((seq + [next_token], new_score))
                
        # Sort candidates by score in descending order
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Keep the top beam_width candidates
        candidates = all_candidates[:beam_width]
        
    return candidates[0][0]
