import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1):
    model.eval()
    for _ in range(max_new_tokens):
        # Crop context if it exceeds model limits
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # REPETITION PENALTY 
        for token_id in set(idx[0].tolist()):
            logits[0, token_id] /= repetition_penalty
        
        # Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Top-P (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens where cumulative prob exceeds top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift right to always keep the first token that crosses threshold
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False  # always keep top token
            
            sorted_logits[sorted_indices_to_remove] = -float('Inf')
            logits = torch.scatter(logits, 1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx