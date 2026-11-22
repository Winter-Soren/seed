"""Independent vectorized checks of the archived E135 one-agent trajectories."""
import numpy as np

import audit_replay_selection as audit


def _previous(initial, carried):
    return np.concatenate((np.asarray(initial)[None], carried[:-1]), axis=0)


def _observe(grid, position, energy, delta):
    offsets = np.arange(-2, 3)
    rows = (position[:, 0, 0, None, None] + offsets[None, :, None]) % 16
    cols = (position[:, 0, 1, None, None] + offsets[None, None, :]) % 16
    ticks = np.arange(len(grid))
    patch = grid[ticks[:, None, None], rows, cols].reshape(-1, 25)
    ones = np.ones_like(energy)
    return np.concatenate((patch == 1, patch == 2, np.zeros((len(grid), 50), np.float32),
                           energy - ones, np.float32(4) * delta, ones), axis=1)[:, None, :]


def verify(data, initial, rules, tape, *, learning=True, script=None, gap=0, masked_first=None):
    exact, require = audit.exact, audit.require
    close = audit.physical_close
    rules = np.asarray(rules, np.int32)
    ticks = np.arange(len(rules))
    exact(data["rule"], rules, "actual payoff schedule without a hidden input")
    exact(data["t"], ticks.astype(np.int32), "native physical clock")
    for name in ("hebbian_enabled", "predictor_enabled"):
        exact(data[name], np.broadcast_to(learning, rules.shape), "registered readout and learning timing " + name)
    audit.encounter.softmax(data["logits"], data["probabilities"])
    exact(data["logits"], np.float32(3) * data["decision_motor"], "native five-way logits")
    require(np.all(np.abs(data["decision_motor"]) <= 1) and np.all(np.abs(data["signal"]) <= 1), "bounded native motor and signal")
    for name, expected in zip(("key_before", "key_after", "action_key", "regrowth_key", "sampled_action", "regrowth_cell"),
                              tape(initial["key"], data["logits"])):
        exact(data[name], np.asarray(expected), "independently reconstructed native " + name)
    actions = data["action"][:, 0]
    require(np.all((actions >= 0) & (actions < 5)), "native five-action range")
    grids = _previous(initial["grid"], data["grid"])
    if script is None:
        require(np.all(data["forced_action"] == -1), "scored actions are all freely sampled")
        exact(data["action"], data["sampled_action"], "each native categorical draw is actually executed")
    else:
        reset = len(script) * 2
        imposed = np.concatenate((script[:, 3:5].reshape(-1), np.zeros(gap, np.int32)))
        exact(data["forced_action"], imposed, "registered balanced physical exposure")
        exact(actions, imposed, "actual imposed replay movement")
        grids[:reset:2] = np.stack([audit.encounter.pair_grid(pair) for pair in script[:, :2]])
        grids[reset] = 0
    exact(data["grid_before"], grids, "complete physical grid before each choice")

    position = _previous(initial["pos"], data["position"])
    exact(data["position_before"], position, "continuous body position")
    energy = _previous(initial["energy"], data["energy"])
    alive = _previous(initial["alive"], data["alive_after"])
    delta = _previous(initial["denergy"], data["denergy"])
    signal = _previous(initial["sig"], data["signal"])
    for name, expected in (("energy_before", energy), ("alive_before", alive),
                           ("incoming_denergy", delta), ("signal_before", signal)):
        close(data[name], expected, "physical body continuity " + name)
    observation = _observe(grids, position, energy, delta)
    if masked_first is not None:
        observation[0, 0, :50] = _observe(masked_first[None], position[:1], energy[:1], delta[:1])[0, 0, :50]
    exact(data["observation"], observation, "only the registered first view masks background A/B features")

    moved = (position + audit.MOVES[actions][:, None] * alive[:, :, None].astype(np.int32)) % 16
    exact(data["position"], moved, "sampled move changes the actual body")
    item = grids[ticks, moved[:, 0, 0], moved[:, 0, 1]][:, None]
    food = (alive > 0) & (item == 1 + rules[:, None])
    poison = (alive > 0) & (item == 2 - rules[:, None])
    exact(data["item"], item, "actual physical destination contents")
    exact(data["food"], food, "food from the full unchanged physical grid")
    exact(data["poison"], poison, "poison from the full unchanged physical grid")
    unclipped = energy - np.float32(.003) * alive + np.float32(.25) * food - np.float32(.25) * poison
    close(data["unclipped_energy"], unclipped, "ordinary energy equation")
    close(data["energy"], np.clip(unclipped, 0, 2), "actual energy clipping")
    close(data["denergy"], (data["energy"] - energy) * alive, "felt clipped energy feedback")
    exact(data["alive_after"], alive * (data["energy"] > 0), "death is absorbing")
    exact(data["efference"], np.eye(5, dtype=np.float32)[data["action"]], "copy the actual executed action")
    if script is not None:
        require(np.all(alive) and np.all((unclipped >= .462 - 1e-5) & (unclipped <= 1.25 + 1e-5)),
                "registered paired exposure keeps every creature alive and unclipped")
        exact(data["energy"], data["unclipped_energy"], "no physical clipping during exposure")

    eat = (food | poison)[:, 0]
    grids[ticks[eat], moved[eat, 0, 0], moved[eat, 0, 1]] = 0
    if script is None:
        cells = data["regrowth_cell"][:, 0]
        grids[ticks[eat], cells[eat, 0], cells[eat, 1]] = item[eat, 0]
    exact(data["grid"], grids, "native consumption and matched resource regrowth")
