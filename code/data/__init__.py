from .dataset import build

def build_dataset(split = 'train', samples = 2000):
    return build(split = split, samples = samples)