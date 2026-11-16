"""AgriSight: plant disease diagnosis that holds up on real field photos,
plus a grounded multilingual advisory assistant.

Heavy dependencies (torch, transformers, ...) are imported lazily inside the
submodules so that lightweight parts (labels, metrics, RAG fusion, grounding)
can be used and tested without a GPU stack.
"""

__version__ = "0.1.0"
