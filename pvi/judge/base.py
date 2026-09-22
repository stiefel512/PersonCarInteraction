"""Judge interface.

An ABC here because the second implementation is the headline ablation, not a
hypothetical: `geometric.py` is the control arm that shows what the VLM adds
over rules alone. If the two arms score the same, the VLM is not earning its
place in the pipeline and the write-up should say so.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from ..propose.rules import Candidate
from ..schema import ClipMeta


@dataclass
class Verdict:
    is_interaction: bool
    type: str
    person_desc: str
    vehicle_desc: str
    note: str
    confidence: float
    # Structured slots from judge/describe.py, when a description pass ran.
    person_attrs: dict[str, Any] | None = None
    vehicle_attrs: dict[str, Any] | None = None


class Judge(abc.ABC):
    @abc.abstractmethod
    def judge(self, cand: Candidate, meta: ClipMeta) -> Verdict | None:
        """Adjudicate one candidate.

        `None` or `is_interaction=False` sends the candidate to the debug
        artifact rather than the deliverable. Returning `None` means "could not
        judge" (e.g. frames unavailable) and is deliberately distinct from a
        confident rejection -- the two should not be counted together when
        diagnosing precision.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier recorded in the output, so a result file states which
        arm produced it."""
