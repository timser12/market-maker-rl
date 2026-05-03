from __future__ import annotations

from dataclasses import dataclass

from mmrl.sim.risk import QuoteProposal


@dataclass(frozen=True, slots=True)
class DiscreteActionMapper:
    base_size_int: int
    normal_offset_ticks: int = 2

    def to_proposal(self, action: int, inventory_sign: int = 0) -> QuoteProposal:
        """Map a discrete research action to quote offsets and sizes."""
        s = self.base_size_int
        if action == 0:  # do nothing
            return QuoteProposal(None, None, 0, 0)
        if action == 1:  # quote both sides tight
            return QuoteProposal(1, 1, s, s)
        if action == 2:  # normal
            return QuoteProposal(self.normal_offset_ticks, self.normal_offset_ticks, s, s)
        if action == 3:  # wide
            return QuoteProposal(5, 5, s, s)
        if action == 4:  # reduce long inventory: smaller bid, larger/tighter ask
            return QuoteProposal(5, 1, max(1, s // 2), s)
        if action == 5:  # reduce short inventory: tighter bid, smaller ask
            return QuoteProposal(1, 5, s, max(1, s // 2))
        if action == 6:  # cancel all
            return QuoteProposal(None, None, 0, 0, cancel_all=True)
        if action == 7:  # only bid
            return QuoteProposal(self.normal_offset_ticks, None, s, 0)
        if action == 8:  # only ask
            return QuoteProposal(None, self.normal_offset_ticks, 0, s)
        raise ValueError(f"unknown action {action}")
