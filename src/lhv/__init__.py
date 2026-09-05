"""Livestock health vision — the spine that carries video through to auditable
per-animal health signals.

The package is organised as one module per stage boundary. Each stage reads
durable records and writes durable records, so any stage can be re-run without
re-running the stage before it:

    ingest -> perception -> identity -> phenotype -> baseline -> events

with ``evaluation`` attaching to the materialised records of every stage.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
