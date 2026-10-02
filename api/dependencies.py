from functools import lru_cache

from src.models.inference import load_bundle


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    return load_bundle()
