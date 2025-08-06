"""
Production-quality beam search implementation using MLX for efficient computation.

This module provides optimized beam search with full MLX acceleration,
including batched logits computation, advanced sampling techniques,
and comprehensive logits/logprobs analysis.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple, Union
from dataclasses import dataclass

from .model import GPTOSSModel
from .config import GPTOSSConfig


@dataclass
class MLXBeamState:
    """MLX-optimized beam state for efficient computation."""
    tokens: mx.array  # Shape: [beam_size, seq_len]
    log_probs: mx.array  # Shape: [beam_size]
    lengths: mx.array  # Shape: [beam_size]
    finished: mx.array  # Shape: [beam_size], boolean mask
    
    def normalized_scores(self, length_penalty: float = 0.6) -> mx.array:
        """Compute length-normalized scores for all beams."""
        # GNMT length penalty: ((5 + length) / (5 + 1)) ** alpha
        penalty = mx.power((5.0 + self.lengths) / 6.0, length_penalty)
        return self.log_probs / penalty


class MLXBeamSearchResult(NamedTuple):
    """Result from MLX beam search with full analysis data."""
    sequences: mx.array  # Shape: [num_beams, max_seq_len]
    scores: mx.array     # Shape: [num_beams]
    logits_history: Optional[List[mx.array]]  # List of [beam_size, vocab_size] arrays
    logprobs_history: Optional[List[mx.array]]  # List of [beam_size, vocab_size] arrays


class MLXProductionBeamSearch:
    """
    Production-quality beam search with full MLX optimization.
    
    Features:
    - Batched computation for all beams simultaneously
    - Efficient top-k sampling with MLX operations
    - Advanced penalties (repetition, n-gram blocking)
    - Comprehensive logits/logprobs tracking
    - Memory-efficient KV caching
    - Configurable stopping criteria
    """
    
    def __init__(self, model: GPTOSSModel, config: GPTOSSConfig):
        """Initialize MLX beam search with model and config."""
        self.model = model
        self.config = config
        self.vocab_size = config.vocab_size
        
    def beam_search(
        self,
        prompt_tokens: Union[List[int], mx.array],
        beam_size: int = 4,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        length_penalty: float = 0.6,
        early_stopping: bool = True,
        return_logits: bool = False,
        return_logprobs: bool = True,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        top_k: int = 50,
        top_p: float = 1.0,
        diversity_penalty: float = 0.0
    ) -> MLXBeamSearchResult:
        """
        Perform beam search with full MLX optimization.
        
        Args:
            prompt_tokens: Initial prompt tokens
            beam_size: Number of beams to maintain
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            length_penalty: Length normalization penalty (GNMT style)
            early_stopping: Stop when all beams finish
            return_logits: Return full logits history
            return_logprobs: Return log probabilities
            eos_token_id: End-of-sequence token
            pad_token_id: Padding token
            repetition_penalty: Penalty for repeated tokens
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            top_k: Top-K filtering
            top_p: Nucleus (top-p) filtering
            diversity_penalty: Penalty for similar beams
            
        Returns:
            MLXBeamSearchResult with sequences, scores, and optional analysis data
        """
        # Convert prompt to MLX array
        if isinstance(prompt_tokens, list):
            prompt_tokens = mx.array(prompt_tokens)
        
        batch_size = 1
        prompt_len = prompt_tokens.shape[0]
        
        # Initialize beam state - replicate prompt for all beams
        initial_tokens = mx.tile(prompt_tokens[None, :], (beam_size, 1))  # [beam_size, prompt_len]
        initial_log_probs = mx.zeros((beam_size,))
        initial_log_probs = initial_log_probs.at[1:].set(-float('inf'))  # Only first beam starts active
        initial_lengths = mx.full((beam_size,), prompt_len, dtype=mx.int32)
        initial_finished = mx.zeros((beam_size,), dtype=mx.bool_)
        
        beam_state = MLXBeamState(
            tokens=initial_tokens,
            log_probs=initial_log_probs,
            lengths=initial_lengths,
            finished=initial_finished
        )
        
        # History tracking
        logits_history = [] if return_logits else None
        logprobs_history = [] if return_logprobs else None
        finished_sequences = []
        
        # KV cache for efficiency
        cache = None
        
        # Generation loop
        for step in range(max_new_tokens):
            # Check if all beams are finished
            if early_stopping and mx.all(beam_state.finished):
                break
            
            # Prepare input for forward pass
            if cache is None:
                # First step: use full sequences
                input_ids = beam_state.tokens  # [beam_size, seq_len]
            else:
                # Subsequent steps: use only last token with cache
                input_ids = beam_state.tokens[:, -1:]  # [beam_size, 1]
            
            # Forward pass through model
            logits, cache = self._compute_logits_batch(input_ids, cache)  # [beam_size, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty_batch(logits, beam_state, repetition_penalty)
            
            # Apply n-gram blocking
            if no_repeat_ngram_size > 0:
                logits = self._apply_ngram_blocking_batch(logits, beam_state, no_repeat_ngram_size)
            
            # Apply top-k filtering
            if top_k > 0:
                logits = self._apply_top_k_filtering(logits, top_k)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                logits = self._apply_top_p_filtering(logits, top_p)
            
            # Compute log probabilities
            log_probs = mx.log_softmax(logits, axis=-1)  # [beam_size, vocab_size]
            
            # Store history
            if return_logits:
                logits_history.append(logits)
            if return_logprobs:
                logprobs_history.append(log_probs)
            
            # Expand beams and select top candidates
            beam_state = self._expand_and_select_beams(
                beam_state, log_probs, beam_size, eos_token_id, 
                max_new_tokens, diversity_penalty
            )
            
            # Move finished beams to results
            finished_mask = beam_state.finished
            if mx.any(finished_mask):
                finished_indices = mx.where(finished_mask)[0]
                for idx in finished_indices:
                    idx_int = int(idx.item())
                    finished_sequences.append({
                        'tokens': beam_state.tokens[idx_int],
                        'score': beam_state.normalized_scores()[idx_int],
                        'log_prob': beam_state.log_probs[idx_int]
                    })
                
                # Early stopping check
                if len(finished_sequences) >= beam_size and early_stopping:
                    break
        
        # Collect final results
        final_sequences = finished_sequences.copy()
        
        # Add any remaining active beams
        active_mask = ~beam_state.finished
        if mx.any(active_mask):
            active_indices = mx.where(active_mask)[0]
            for idx in active_indices:
                idx_int = int(idx.item())
                final_sequences.append({
                    'tokens': beam_state.tokens[idx_int],
                    'score': beam_state.normalized_scores()[idx_int],
                    'log_prob': beam_state.log_probs[idx_int]
                })
        
        # Sort by score and take top beam_size
        final_sequences.sort(key=lambda x: float(x['score'].item()), reverse=True)
        final_sequences = final_sequences[:beam_size]
        
        # Convert to arrays
        max_len = max(seq['tokens'].shape[0] for seq in final_sequences)
        sequences = mx.zeros((beam_size, max_len), dtype=mx.int32)
        scores = mx.zeros((beam_size,))
        
        for i, seq_data in enumerate(final_sequences):
            tokens = seq_data['tokens']
            sequences = sequences.at[i, :tokens.shape[0]].set(tokens)
            scores = scores.at[i].set(seq_data['score'])
        
        return MLXBeamSearchResult(
            sequences=sequences,
            scores=scores,
            logits_history=logits_history,
            logprobs_history=logprobs_history
        )
    
    def _compute_logits_batch(self, input_ids: mx.array, cache: Optional[List] = None) -> Tuple[mx.array, List]:
        """Compute logits for a batch of sequences efficiently."""
        # Forward pass through the model
        logits, new_cache = self.model(input_ids, cache=cache)
        
        # Extract logits for the last token of each sequence
        if logits.ndim == 3:  # [batch_size, seq_len, vocab_size]
            logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        return logits, new_cache
    
    def _apply_repetition_penalty_batch(self, logits: mx.array, beam_state: MLXBeamState, penalty: float) -> mx.array:
        """Apply repetition penalty to all beams simultaneously using vectorized operations."""
        if penalty == 1.0:
            return logits
        
        # Vectorized repetition penalty computation
        beam_size, seq_len = beam_state.tokens.shape
        penalty_mask = mx.zeros_like(logits)  # [beam_size, vocab_size]
        
        # For each position in vocabulary, count occurrences across all beams
        for vocab_id in range(min(2000, self.vocab_size)):  # Process in chunks for memory efficiency
            # Count occurrences of this token across all beam sequences
            token_matches = (beam_state.tokens == vocab_id)  # [beam_size, seq_len]
            token_counts = mx.sum(token_matches, axis=1)  # [beam_size]
            
            # Apply penalty based on count
            penalty_factors = mx.where(
                logits[:, vocab_id] > 0,
                1.0 / (penalty ** token_counts),
                penalty ** token_counts
            )
            penalty_mask = penalty_mask.at[:, vocab_id].set(penalty_factors)
        
        return logits * penalty_mask
    
    def _apply_ngram_blocking_batch(self, logits: mx.array, beam_state: MLXBeamState, ngram_size: int) -> mx.array:
        """Apply n-gram blocking using vectorized operations."""
        if ngram_size <= 1:
            return logits
        
        beam_size = beam_state.tokens.shape[0]
        blocked_logits = logits.copy()
        
        # Vectorized n-gram blocking for efficiency
        for beam_idx in range(beam_size):
            seq_len = int(beam_state.lengths[beam_idx].item())
            if seq_len < ngram_size:
                continue
                
            beam_tokens = beam_state.tokens[beam_idx, :seq_len]
            context = beam_tokens[-ngram_size+1:] if ngram_size > 1 else mx.array([])
            
            # Find matching n-gram contexts efficiently
            if context.shape[0] > 0:
                # Create sliding window of n-grams
                for i in range(seq_len - ngram_size + 1):
                    ngram_context = beam_tokens[i:i + ngram_size - 1]
                    if mx.array_equal(ngram_context, context):
                        blocked_token = int(beam_tokens[i + ngram_size - 1].item())
                        if 0 <= blocked_token < self.vocab_size:
                            blocked_logits = blocked_logits.at[beam_idx, blocked_token].set(-float('inf'))
        
        return blocked_logits
    
    def _apply_top_k_filtering(self, logits: mx.array, top_k: int) -> mx.array:
        """Apply top-k filtering to logits."""
        if top_k <= 0 or top_k >= self.vocab_size:
            return logits
        
        # Find the k-th largest value for each beam
        topk_values = mx.topk(logits, k=top_k, axis=-1)
        threshold = topk_values[1][:, -1:]  # [beam_size, 1]
        
        # Mask values below threshold
        mask = logits >= threshold
        return mx.where(mask, logits, -float('inf'))
    
    def _apply_top_p_filtering(self, logits: mx.array, top_p: float) -> mx.array:
        """Apply nucleus (top-p) filtering to logits."""
        if top_p >= 1.0:
            return logits
        
        # Sort logits in descending order
        sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]  # [beam_size, vocab_size]
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        
        # Compute cumulative probabilities
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        
        # Find cutoff indices
        cutoff_mask = cumulative_probs <= top_p
        cutoff_mask = cutoff_mask.at[:, 0].set(True)  # Always keep at least one token
        
        # Create filtered logits
        filtered_sorted_logits = mx.where(cutoff_mask, sorted_logits, -float('inf'))
        
        # Restore original order
        filtered_logits = mx.zeros_like(logits)
        for beam_idx in range(logits.shape[0]):
            for pos_idx in range(logits.shape[1]):
                orig_idx = sorted_indices[beam_idx, pos_idx]
                filtered_logits = filtered_logits.at[beam_idx, orig_idx].set(
                    filtered_sorted_logits[beam_idx, pos_idx]
                )
        
        return filtered_logits
    
    def _expand_and_select_beams(
        self, 
        beam_state: MLXBeamState, 
        log_probs: mx.array, 
        beam_size: int,
        eos_token_id: int,
        max_length: int,
        diversity_penalty: float
    ) -> MLXBeamState:
        """Vectorized beam expansion and selection for maximum MLX efficiency."""
        batch_size = beam_state.tokens.shape[0]
        active_mask = ~beam_state.finished
        
        if not mx.any(active_mask):
            return beam_state
        
        # Vectorized candidate generation
        # Shape: [beam_size, vocab_size]
        expanded_scores = beam_state.log_probs[:, None] + log_probs
        
        # Mask inactive beams
        expanded_scores = mx.where(
            active_mask[:, None], 
            expanded_scores, 
            -float('inf')
        )
        
        # Flatten and get top candidates
        flat_scores = expanded_scores.reshape(-1)
        flat_indices = mx.arange(flat_scores.shape[0])
        
        # Get top beam_size * 2 candidates for diversity
        top_k = min(beam_size * 4, flat_scores.shape[0])
        top_scores, top_flat_indices = mx.topk(flat_scores, k=top_k)
        
        # Convert flat indices back to (beam_idx, token_id)
        beam_indices = top_flat_indices // self.vocab_size
        token_indices = top_flat_indices % self.vocab_size
        
        # Apply diversity penalty efficiently
        if diversity_penalty > 0:
            unique_tokens, inverse_indices = mx.unique(token_indices, return_inverse=True)
            token_counts = mx.bincount(inverse_indices, minlength=len(unique_tokens))
            diversity_penalties = diversity_penalty * (token_counts[inverse_indices] - 1)
            top_scores = top_scores - diversity_penalties
        
        # Re-sort after diversity penalty
        if diversity_penalty > 0:
            reranked_indices = mx.argsort(top_scores)[::-1]
            top_scores = top_scores[reranked_indices]
            beam_indices = beam_indices[reranked_indices]
            token_indices = token_indices[reranked_indices]
        
        # Select final beam_size candidates
        selected_beam_indices = beam_indices[:beam_size]
        selected_token_indices = token_indices[:beam_size]
        selected_scores = top_scores[:beam_size]
        
        # Build new sequences efficiently
        max_seq_len = beam_state.tokens.shape[1] + 1
        new_tokens = mx.zeros((beam_size, max_seq_len), dtype=mx.int32)
        new_log_probs = mx.zeros(beam_size)
        new_lengths = mx.zeros(beam_size, dtype=mx.int32)
        new_finished = mx.zeros(beam_size, dtype=mx.bool_)
        
        for i in range(beam_size):
            beam_idx = int(selected_beam_indices[i].item())
            token_id = int(selected_token_indices[i].item())
            
            # Copy old sequence and append new token
            old_seq_len = int(beam_state.lengths[beam_idx].item())
            new_tokens = new_tokens.at[i, :old_seq_len].set(beam_state.tokens[beam_idx, :old_seq_len])
            new_tokens = new_tokens.at[i, old_seq_len].set(token_id)
            
            new_log_probs = new_log_probs.at[i].set(selected_scores[i])
            new_lengths = new_lengths.at[i].set(old_seq_len + 1)
            
            # Check if finished
            finished = (token_id == eos_token_id) or (old_seq_len + 1 >= max_length)
            new_finished = new_finished.at[i].set(finished)
        
        return MLXBeamState(
            tokens=new_tokens,
            log_probs=new_log_probs,
            lengths=new_lengths,
            finished=new_finished
        )
    
    def _apply_diversity_penalty(
        self, 
        scores: mx.array, 
        tokens: mx.array, 
        beam_indices: mx.array, 
        penalty: float
    ) -> mx.array:
        """Apply diversity penalty to encourage different outputs across beams."""
        # Group candidates by beam and apply penalty for similar tokens
        adjusted_scores = scores
        
        # Simple implementation: penalize tokens that are already being considered by other beams
        unique_tokens, token_counts = mx.unique(tokens, return_counts=True)
        
        for i, token in enumerate(tokens):
            token_count = token_counts[mx.where(unique_tokens == token)[0][0]]
            if token_count > 1:
                adjusted_scores = adjusted_scores.at[i].set(
                    adjusted_scores[i] - penalty * (token_count - 1)
                )
        
        return adjusted_scores