"""Evaluation helpers; import numerical diagnostics only when requested."""

from importlib import import_module

__all__ = [
    "cosine_similarity_by_event_level",
    "joint_pca_rgb",
    "temporal_feature_stability",
    "token_similarity_map",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".diagnostics", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

