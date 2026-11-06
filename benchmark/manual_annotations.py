"""Selection of independent, human-validated ground truth.

Annotation JSON may also contain automatic proposals and drafts. Those must
never contribute to benchmark reference counts or curve-quality scores.
"""

from __future__ import annotations


def validated_manual_chirps(record):
    """Return only explicitly validated, manually sourced chirps."""
    return [
        chirp for chirp in record.get("chirps", [])
        if isinstance(chirp, dict)
        and chirp.get("source") == "manual"
        and chirp.get("status") == "validated"
    ]
