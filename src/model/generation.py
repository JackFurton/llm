import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k: keep only top k tokens with highest probability (top-k filtering)
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering)
        filter_value: value to assign to filtered tokens
    
    Returns:
        filtered logits
    """
    assert logits.dim() == 2  # batch size x vocabulary size
    
    # Apply top-k filtering
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits

def generate_with_sampling(
    model,
    input_ids,
    max_length,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    pad_token_id=0,
    eos_token_id=None
):
    """
    Generate text using advanced sampling strategies
    
    Args:
        model: The language model
        input_ids: Input token IDs (batch_size, seq_len)
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (0 to disable)
        top_p: Top-p (nucleus) sampling parameter (0 to disable)
        repetition_penalty: Penalty for repeating tokens
        pad_token_id: ID of padding token
        eos_token_id: ID of end-of-sequence token (None to disable)
    
    Returns:
        Generated token IDs (batch_size, seq_len)
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Create attention mask (causal/autoregressive)
    input_ids_len = input_ids.shape[1]
    
    # Keep track of which sequences are already finished
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
    
    # Clone input_ids to avoid modifying the original
    output_ids = input_ids.clone()
    
    # Generate tokens
    for step in range(max_length - input_ids_len):
        # Forward pass
        with torch.no_grad():
            # Create attention mask for current sequence length
            seq_len = output_ids.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
            
            # Get logits
            logits = model(output_ids, mask=mask)
            
            # Only use the last token's logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(output_ids[i].tolist()):
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply filtering
            filtered_logits = top_k_top_p_filtering(
                next_token_logits.clone(),
                top_k=top_k,
                top_p=top_p
            )
            
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # If a sequence is finished, replace next token with pad token
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # Append next tokens to output
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update which sequences are finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id).long())
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
    
    return output_ids

def generate_with_beam_search(
    model,
    input_ids,
    max_length,
    beam_size=5,
    pad_token_id=0,
    eos_token_id=None,
    length_penalty=1.0
):
    """
    Generate text using beam search
    
    Args:
        model: The language model
        input_ids: Input token IDs (batch_size, seq_len)
        max_length: Maximum length to generate
        beam_size: Beam size
        pad_token_id: ID of padding token
        eos_token_id: ID of end-of-sequence token (None to disable)
        length_penalty: Length penalty (>1.0 favors longer sequences, <1.0 favors shorter)
    
    Returns:
        Generated token IDs (batch_size, seq_len)
    """
    # Only support batch size 1 for simplicity
    assert input_ids.shape[0] == 1, "Beam search only supports batch size 1"
    
    device = input_ids.device
    batch_size = 1
    vocab_size = model.output_projection.out_features
    
    # Expand input to beam size
    input_ids = input_ids.repeat(beam_size, 1)
    
    # Scores for each beam
    beam_scores = torch.zeros(beam_size, device=device)
    beam_scores[1:] = -1e9  # Only the first beam is active at the start
    
    # Keep track of finished beams
    done = [False for _ in range(beam_size)]
    
    # Generate tokens
    for step in range(max_length - input_ids.shape[1]):
        # Forward pass
        with torch.no_grad():
            # Create attention mask for current sequence length
            seq_len = input_ids.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
            
            # Get logits
            logits = model(input_ids, mask=mask)
            
            # Only use the last token's logits
            next_token_logits = logits[:, -1, :]
            
            # Calculate log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (beam_size, vocab_size)
            
            # Add beam scores to token scores
            next_token_scores = next_token_scores + beam_scores[:, None]  # (beam_size, vocab_size)
            
            # Reshape for easier handling
            next_scores = next_token_scores.view(batch_size, beam_size * vocab_size)  # (1, beam_size * vocab_size)
            
            # Get the top-k next tokens and their scores
            next_scores, next_tokens = torch.topk(next_scores, 2 * beam_size, dim=1)
            
            # Convert to beam indices and token indices
            next_beam_indices = next_tokens // vocab_size
            next_token_indices = next_tokens % vocab_size
            
            # Initialize next beams
            next_beam_scores = torch.zeros(beam_size, device=device)
            next_beam_tokens = torch.zeros(beam_size, 1, dtype=torch.long, device=device)
            next_beam_indices = torch.zeros(beam_size, dtype=torch.long, device=device)
            
            # Fill next beams
            beam_idx = 0
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_token_indices[0], next_scores[0])
            ):
                beam_id = next_beam_indices[0, beam_token_rank]
                
                # Skip finished beams
                if done[beam_id]:
                    continue
                
                if beam_idx >= beam_size:
                    break
                
                next_beam_scores[beam_idx] = beam_token_score
                next_beam_tokens[beam_idx, 0] = beam_token_id
                next_beam_indices[beam_idx] = beam_id
                beam_idx += 1
            
            # Check if we're done
            for beam_id in range(beam_size):
                if next_beam_tokens[beam_id, 0] == eos_token_id:
                    done[beam_id] = True
            
            if all(done):
                break
            
            # Update beam scores
            beam_scores = next_beam_scores
            
            # Reorder beams
            input_ids = torch.cat([input_ids[next_beam_indices], next_beam_tokens], dim=1)
    
    # Return the highest scoring beam
    return input_ids[0:1]
