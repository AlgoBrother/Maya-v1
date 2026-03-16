import torch
import mayatok_bpe as bpe
from generate import generate
from model import MayaTransformer
from config import MayaConfig


# ── Load model ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = MayaConfig()
model = MayaTransformer(config)

checkpoint = torch.load("/mnt/d/Maya_checkpoints/best.pt", 
                        map_location=device, weights_only=False)
state_dict = checkpoint['model_state_dict']
unwrapped = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(unwrapped)
model.to(device)
model.eval()
print("Maya loaded ✅")

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = bpe.PyBPETokenizer.load("bpe_tokenizer_py.json")

# ── Generate ─────────────────────────────────────────────────────────────────
prompts = [
    "The capital of France is",
    "Once upon a time",
    "Machine learning is",
]

for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    out = generate(model, idx, max_new_tokens=100)
    decoded = tokenizer.decode(out[0].tolist())
    
    print(f"\nPrompt: {prompt}")
    print(f"Output: {decoded}")
    print("-" * 60)