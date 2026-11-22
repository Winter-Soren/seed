"""Descriptive counts from already verified ordinary solo traces, one value per life."""

import numpy as np


# The centered 5-by-5 patch is flattened in row-major order.
# The five actions are stay, north, south, west, east.
DESTINATIONS = (12, 7, 17, 11, 13)
FIELDS = (
    "first_feedback_gain", "first_feedback_loss",
    "reachable_item_after_feedback", "both_colors_after_feedback",
    "second_contact", "both_colors_at_second_contact",
    "repeat_after_gain", "switch_after_gain", "repeat_after_loss", "switch_after_loss",
)


def feedback_opportunities(observation, action, alive_before):
    """Count opportunities from feedback arrival through the next contact, inclusive.

    Accepts arrays ending in (turn, 1, 103), (turn, 1), (turn, 1).
    Leading axes index assigned lives. Only visible destination type and the
    next live observation's full, unclipped energy change identify feedback.
    Missing feedback, opportunities and later contacts contribute zeros.
    """
    obs, moves, live = map(np.asarray, (observation, action, alive_before))
    if obs.ndim < 3 or obs.shape[-2:] != (1, 103) or not obs.shape[-3]:
        raise ValueError("expected the native single-agent observation")
    if moves.shape != obs.shape[:-1] or live.shape != moves.shape:
        raise ValueError("turn and assignment axes must agree")
    if moves.dtype.kind not in "iu" or np.any((moves < 0) | (moves >= 5)):
        raise ValueError("invalid native action index")
    if not np.isin(live, (0, 1)).all():
        raise ValueError("invalid live indicator")
    leading, turns = obs.shape[:-3], obs.shape[-3]
    obs = obs.reshape(-1, turns, 103)
    moves = moves.reshape(-1, turns)
    live = live.reshape(-1, turns).astype(bool)
    if np.any(live[:, 1:] & ~live[:, :-1]):
        raise ValueError("absorbing death was reversed")
    a = obs[..., list(DESTINATIONS)]
    b = obs[..., [25 + i for i in DESTINATIONS]]
    if (not np.isin(a, (0, 1)).all() or not np.isin(b, (0, 1)).all()
            or np.any(a * b) or not np.isfinite(obs[..., 100:102]).all()):
        raise ValueError("invalid visible item planes or energy observation")
    chosen = np.take_along_axis(a + 2*b, moves[..., None], axis=-1)[..., 0]
    contacts = live & (chosen != 0)
    sees_a, sees_b = np.any(a, axis=-1), np.any(b, axis=-1)
    flags = {name: np.zeros(len(obs), dtype=bool) for name in FIELDS}
    for life in range(len(obs)):
        hits = np.flatnonzero(contacts[life])
        if not len(hits):
            continue
        first = int(hits[0])
        feedback = first + 1
        if feedback == turns or not live[life, feedback]:
            continue
        energy = 1.0 + float(obs[life, feedback, 100])
        change = float(obs[life, feedback, 101]) / 4
        if not 0 < energy < 2 or abs(abs(change + .003) - .25) > 1e-6:
            continue
        sign = "gain" if change > 0 else "loss"
        flags["first_feedback_" + sign][life] = True
        stop = int(hits[1]) + 1 if len(hits) > 1 else turns
        window = slice(feedback, stop)
        active = live[life, window]
        left, right = sees_a[life, window], sees_b[life, window]
        flags["reachable_item_after_feedback"][life] = np.any(active & (left | right))
        flags["both_colors_after_feedback"][life] = np.any(active & left & right)
        if len(hits) > 1:
            second = int(hits[1])
            flags["second_contact"][life] = True
            flags["both_colors_at_second_contact"][life] = sees_a[life, second] & sees_b[life, second]
            choice = "repeat" if chosen[life, second] == chosen[life, first] else "switch"
            flags[choice + "_after_" + sign][life] = True
    return {name: values.reshape(leading) for name, values in flags.items()}
