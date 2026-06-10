import re

def safe_decode(tokenizer, token_ids, space_token_id=292):
    parts = []
    for tid in token_ids:
        if tid == space_token_id:
            parts.append(' ')
        else:
            tok = tokenizer.decode([tid])
            parts.append(tok)
    
    text = "".join(parts)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()
# ----------------- USER INPUT VERSION ----------------- 
import torch
import mayatok as bpe
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
tokenizer = bpe.PyBPETokenizer.load(r"../../mayatok_vocab.json")

eos_token_id = 289  # Hardcoded EOS token ID for Maya [I am fixing this in V2]

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
    generated_tokens = out[0][len(tokens):].tolist()
    
    # # DEBUG BLOCK [COMMENTED OUT]
    # print("Raw tokens:", generated_tokens[:30])
    # print("Unique tokens:", sorted(set(generated_tokens[:30])))
    # for tid in generated_tokens[:30]:
    #     print(tid, repr(tokenizer.decode([tid])))

    # Strip EOS and any control tokens
    SKIP_TOKENS = {289, 288, 2451}  # EOS, BOS — add others after checking above
    clean_tokens = [t for t in generated_tokens if t not in SKIP_TOKENS]

    decoded = safe_decode(tokenizer, clean_tokens)
    print(f"\nMaya: {decoded}")
    print("-" * 50)