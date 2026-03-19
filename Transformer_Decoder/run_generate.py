# import torch
# import mayatok_bpe as bpe
# from generate import generate
# from model import MayaTransformer
# from config import MayaConfig


# # ── Load model ──────────────────────────────────────────────────────────────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = MayaConfig()
# model = MayaTransformer(config)

# checkpoint = torch.load(r"D:\Maya_checkpoints\best.pt", 
#                         map_location=device, weights_only=False)
# state_dict = checkpoint['model_state_dict']
# unwrapped = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
# model.load_state_dict(unwrapped)
# model.to(device)
# model.eval()
# print("Maya loaded ✅")

# # ── Tokenizer ────────────────────────────────────────────────────────────────
# tokenizer = bpe.PyBPETokenizer.load("bpe_tokenizer_py.json")
# # ── Generate ─────────────────────────────────────────────────────────────────
# prompts = [
#     "The capital of France is",
#     "Once upon a time",
#     "Machine learning is",
#     "2 + 2 = 4, but 2 + 3 =",
#     "The largest planet is",
#     "Python is a programming",
#     "The meaning of life is",
#     "Compare minimum spanning tree algorithm: Prim's vs Kruskal's"
# ]

# for prompt in prompts:
#     tokens = tokenizer.encode(prompt)
#     idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
#     out = generate(model, idx, max_new_tokens=100)
#     decoded = tokenizer.decode(out[0].tolist())
    
#     print(f"\nPrompt: {prompt}")
#     print(f"Output: {decoded}")
#     print("-" * 60)

# ----------------- USER INPUT VERSION ----------------- 
import torch
import mayatok_bpe as bpe
from generate import generate
from model import MayaTransformer
from config import MayaConfig

# ── Load model ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = MayaConfig()
model = MayaTransformer(config)

checkpoint = torch.load(
    r"D:\Maya_checkpoints\best.pt",
    map_location=device
)

state_dict = checkpoint['model_state_dict']
unwrapped = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model.load_state_dict(unwrapped)
model.to(device)
model.eval()

print("Maya loaded ✅")

# ── Tokenizer ──────────────────────────────────────────────
tokenizer = bpe.PyBPETokenizer.load("C:\\Users\\Ashwin Rajhans\\Maya-v1\\Transformer_Decoder\\bpe_tokenizer_py.json")

# OPTIONAL [Get EOS token ID if needed for generation stopping criteria]*
eos_token_id = 289  # Hardcoded EOS token ID for Maya (adjust if your tokenizer uses a different ID)

# ── Chat loop ──────────────────────────────────────────────
while True:
    user_input = input("Enter: ")

    if user_input.lower() == "exit":
        break

    tokens = [288] + tokenizer.encode(user_input)

    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    idx = idx[:, -config.block_size:]

    with torch.no_grad():
        out = generate(model, idx, max_new_tokens=100)

    # Only decode generated part
    generated_tokens = out[0][len(tokens):]
    decoded = tokenizer.decode(generated_tokens.tolist())

    print(f"\nMaya: {decoded}")
    print("-" * 500)