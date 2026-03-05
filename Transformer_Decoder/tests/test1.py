import torch
from model import MayaTransformer # Assuming your model is here
from config import MayaConfig

def audit_vram(config):
    model = MayaTransformer(config)
    
    # 1. Parameter Count
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Memory Breakdown (Assuming bf16/fp16 for training)
    # Weights (2 bytes per param)
    model_mem = params * 2 / (1024**2) 
    
    # Gradients (2 bytes per param)
    grad_mem = trainable_params * 2 / (1024**2)
    
    # Optimizer States (AdamW keeps 2 buffers in fp32 = 8 bytes per param)
    opt_mem = trainable_params * 8 / (1024**2)
    
    # 3. Activation Estimate (Roughly Batch * Seq * Hidden * Layers * 2)
    # Using your MayaConfig: 4 * 256 * 512 * 12 * 2 bytes
    act_mem = (4 * config.block_size * config.n_embd * config.n_layers * 2) / (1024**2)

    total_est = model_mem + grad_mem + opt_mem + act_mem + 1024 # +1GB for CUDA overhead
    
    print(f"--- 🚀 Maya-V1 Audit ---")
    print(f"Total Parameters: {params/1e6:.2f}M")
    print(f"Model Weights:    {model_mem:.2f} MB")
    print(f"Gradients:        {grad_mem:.2f} MB")
    print(f"Optimizer States: {opt_mem:.2f} MB")
    print(f"Activations (est): {act_mem:.2f} MB")
    print(f"------------------------")
    print(f"TOTAL ESTIMATED:  {total_est/1024:.2f} GB")
    
    if total_est / 1024 > 7.0: # Leaving 1GB for Windows/System
        print("⚠️ DANGER: You are close to the 8GB limit. Consider lowering n_embd.")
    else:
        print("✅ SAFE: This should run smoothly on your 4070.")

if __name__ == "__main__":
    audit_vram(MayaConfig())