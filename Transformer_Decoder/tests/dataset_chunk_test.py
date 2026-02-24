from Transformer_Decoder.dataloader import get_dataloader
from Transformer_Decoder.config import MayaConfig

def main():
    loader = get_dataloader("D:\\MAYADataset\\chunkfiles\\chunks", MayaConfig())
    x, y = next(iter(loader))
    print(f"Batch Shape: {x.shape}")
    print(f"First 10 tokens: {x[0, :10]}")

if __name__ == "__main__":
    main()