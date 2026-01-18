# iopaint/download/exceptions.py

class ModelScanError(Exception):
    """Base exception for model scanning errors."""
    pass


class CorruptedModelError(ModelScanError):
    """Model file is corrupted or unreadable."""
    pass


class UnsupportedFormatError(ModelScanError):
    """Model format is not supported."""
    pass


class CacheReadError(ModelScanError):
    """Failed to read cache file."""
    pass


class ModelIncompatibleError(ModelScanError):
    """Model is incompatible with current system."""
    pass


class NetworkError(ModelScanError):
    """Network-related error during model download."""
    pass