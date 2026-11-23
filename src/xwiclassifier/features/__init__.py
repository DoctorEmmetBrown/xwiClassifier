"""Extraction de features (moteur xwi et reproduction des features ilastik)."""
from .engine import (  # noqa: F401
    DEFAULT_FAMILIES, DEFAULT_RADII, DEFAULT_SCALES, FAMILIES, MORPHO_FAMILIES,
    SMOOTH_FAMILIES, FeatureEngine, FeatureSpec, register_family,
)


def make_extractor(features=None, n_threads=None):
    """Construit un extracteur depuis une FeatureSpec, un dict serialise ou un extracteur."""
    if features is None:
        return FeatureEngine(FeatureSpec(), n_threads=n_threads)
    if isinstance(features, FeatureSpec):
        return FeatureEngine(features, n_threads=n_threads)
    if isinstance(features, dict):
        kind = features.get("type", "xwi")
        if kind == "xwi":
            return FeatureEngine(FeatureSpec.from_dict(features), n_threads=n_threads)
        if kind == "ilastik":
            from .ilastik import IlastikFeatures
            return IlastikFeatures.from_dict(features)
        raise ValueError(f"Type d'extracteur inconnu : {kind}")
    if hasattr(features, "compute") and hasattr(features, "halo"):
        return features
    raise TypeError(f"Features non reconnues : {type(features)}")
