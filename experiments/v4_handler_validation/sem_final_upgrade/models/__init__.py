from .cnn_bigru import CNNBiGRUSegmenter
from .inherited_tcn import InheritedTCNSegmenter


def build_model(name: str):
    if name == "inherited_tcn":
        return InheritedTCNSegmenter()
    if name == "cnn_bigru":
        return CNNBiGRUSegmenter()
    raise KeyError(f"Unknown SEM architecture: {name}")


__all__ = ["CNNBiGRUSegmenter", "InheritedTCNSegmenter", "build_model"]
