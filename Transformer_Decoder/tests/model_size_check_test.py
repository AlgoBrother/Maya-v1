import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import MayaTransformer
from config import MayaConfig

config = MayaConfig()
model = MayaTransformer(config)

print(model.transformer.h[0].mlp.w1.weight.shape)
