import torch
import torch.nn.functional as F



@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    model.eval()
    for _ in range(max_new_tokens):
        # Crop context if it exceeds model limits
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Top-K filtering
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx