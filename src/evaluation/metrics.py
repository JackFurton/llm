import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter

def calculate_perplexity(model, dataloader, device="cpu"):
    """
    Calculate perplexity of a model on a dataset.
    Perplexity = exp(average cross-entropy loss)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            # Calculate cross entropy loss
            loss = F.cross_entropy(outputs, labels, ignore_index=0, reduction="sum")
            
            total_loss += loss.item()
            total_tokens += (labels != 0).sum().item()
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    
    return perplexity

def calculate_accuracy(model, dataloader, device="cpu"):
    """
    Calculate token prediction accuracy on a dataset
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids)
            predictions = outputs.argmax(dim=-1)
            
            # Only count non-padding tokens
            mask = (labels != 0)
            correct += ((predictions == labels) & mask).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def calculate_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for generated text.
    
    Args:
        references: List of reference token sequences
        hypotheses: List of generated token sequences
        max_n: Maximum n-gram size to consider
    
    Returns:
        BLEU score (0-1)
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    # Calculate n-gram precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        
        for ref_tokens, hyp_tokens in zip(references, hypotheses):
            # Count n-grams in hypothesis
            hyp_ngrams = Counter()
            for i in range(len(hyp_tokens) - n + 1):
                ngram = tuple(hyp_tokens[i:i+n])
                hyp_ngrams[ngram] += 1
            
            # Count matching n-grams in reference
            ref_ngrams = Counter()
            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i+n])
                ref_ngrams[ngram] += 1
            
            # Count matches
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams[ngram])
            
            total += max(1, len(hyp_tokens) - n + 1)
        
        # Calculate precision for this n
        precision = matches / total if total > 0 else 0
        precisions.append(precision)
    
    # Calculate brevity penalty
    total_ref_len = sum(len(ref) for ref in references)
    total_hyp_len = sum(len(hyp) for hyp in hypotheses)
    bp = math.exp(min(0, 1 - total_ref_len / total_hyp_len)) if total_hyp_len > 0 else 0
    
    # Calculate final BLEU score
    if all(p > 0 for p in precisions):
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        bleu = 0
    
    return bleu

def evaluate_model(model, tokenizer, test_texts, device="cpu", max_length=50, temperature=0.8):
    """
    Comprehensive evaluation of a language model
    
    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for encoding/decoding text
        test_texts: List of test texts
        device: Device to run evaluation on
        max_length: Maximum length for generation
        temperature: Temperature for sampling
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    results = {}
    
    # Prepare prompts and references
    prompts = []
    references = []
    
    for text in test_texts:
        # Use first 20% of each text as prompt
        split_point = max(10, int(len(text) * 0.2))
        prompt = text[:split_point]
        reference = text[split_point:split_point + max_length]
        
        prompts.append(prompt)
        references.append(reference)
    
    # Generate continuations
    generations = []
    for prompt in prompts:
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids, 
                max_length=prompt_ids.size(1) + max_length,
                temperature=temperature
            )
        
        # Extract only the generated part (excluding the prompt)
        generated_text = tokenizer.decode(generated_ids[0][prompt_ids.size(1):].tolist())
        generations.append(generated_text)
    
    # Tokenize references and generations for BLEU calculation
    tokenized_refs = [[c for c in ref] for ref in references]
    tokenized_gens = [[c for c in gen] for gen in generations]
    
    # Calculate BLEU score
    results["bleu"] = calculate_bleu(tokenized_refs, tokenized_gens)
    
    # Calculate diversity metrics
    all_tokens = []
    bigrams = []
    
    for gen in tokenized_gens:
        all_tokens.extend(gen)
        for i in range(len(gen) - 1):
            bigrams.append((gen[i], gen[i+1]))
    
    unique_tokens = len(set(all_tokens)) if all_tokens else 0
    unique_bigrams = len(set(bigrams)) if bigrams else 0
    
    results["unique_token_ratio"] = unique_tokens / len(all_tokens) if all_tokens else 0
    results["unique_bigram_ratio"] = unique_bigrams / len(bigrams) if bigrams else 0
    
    # Return all metrics
    return results
