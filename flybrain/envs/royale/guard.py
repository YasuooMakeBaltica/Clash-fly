"""Coach guard: the fly brain decides, coach rules step in where it is weak.

The brain picks a card role and a lane; :func:`pick_card` turns that into a
card. Then the guard looks at what the coach would do and why:

* defend / finish a tower: the coach's move wins. These are timing- and
  counter-sensitive and the brain gets them wrong most often.
* the brain waits while the coach sees a good chance (a positive spell trade,
  a counter-push, punishing an enemy who just spent elixir, or elixir about
  to be wasted): the coach's move is played.
* the brain casts a spell with nothing worth hitting: it waits instead.
* the brain plays a card while the coach also wants to play: the brain's card
  goes to the coach's lane (the coach knows which lane needs it).

Everything else is the brain's own call.
"""

from __future__ import annotations

from .db import ROLES
from .env import pick_card
from .strategy import Coach, View, spell_spot

TAKE_OVER = ("finish", "defend")
WHEN_IDLE = ("trade", "counterpush", "punish", "leak")


def guard(view: View, card: str | None, lane: int, coach: Coach | None = None) -> tuple[str | None, int, str]:
    """(card or None, lane, who): who is 'fly', 'coach:<reason>' or 'veto'."""
    coach = coach or Coach()
    c_card, c_lane, reason = coach.plan(view)
    if reason in TAKE_OVER and c_card is not None:
        return c_card, c_lane, f"coach:{reason}"
    if card is None:
        if reason in WHEN_IDLE and c_card is not None:
            return c_card, c_lane, f"coach:{reason}"
        return None, lane, "fly"
    c = view.db.cards[card]
    if c.type == "spell" and not c.summons and card != c_card and spell_spot(view, card)[0] < c.elixir:
        return None, lane, "veto"
    if c_card is not None:
        lane = c_lane
    return card, lane, "fly"


def guarded_action(view: View, role_index: int, lane: int, coach: Coach | None = None):
    """Brain action (role index, 0 = wait; lane) -> (card or None, lane, who)."""
    card = pick_card(view, ROLES[role_index - 1]) if role_index > 0 else None
    return guard(view, card, lane, coach)
