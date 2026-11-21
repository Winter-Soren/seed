"""E121 fixed readers of public demonstration pixels, with no fitted parameters."""
from numbers import Integral

import numpy as np


PROTOCOL = "social-channel-v1"
ARMS = ("observed", "yoked", "no_demo", "scene_only")
METHODS = ("track", "scene")
MOVES = np.array(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)), np.int32)
SOUTH = np.array((1, 0), np.int32)


def _pixels(frames, radius):
    frames = np.asarray(frames)
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 1:
        raise ValueError("radius must be a positive integer")
    side, area = 2 * int(radius) + 1, (2 * int(radius) + 1) ** 2
    if (frames.ndim != 2 or frames.shape[0] < 3 or frames.shape[1] != 4 * area + 3
            or frames.dtype.kind not in "iuf" or not np.isfinite(frames).all()):
        raise ValueError("movie must contain finite local visual observations")
    visual = frames[:, :3 * area].reshape(len(frames), 3, side, side)
    if not np.isin(visual, (0, 1)).all() or np.any(visual[:, 0] + visual[:, 1] > 1):
        raise ValueError("item and occupancy channels must be binary with exclusive item colors")
    if np.any(visual[:, 2].sum((1, 2)) > 1):
        raise ValueError("a public movie may contain at most one demonstrator")
    return visual[:, :2].astype(bool), visual[:, 2].astype(bool)


def _finish(colors, identified):
    votes = np.array([np.count_nonzero(colors == color) for color in (0, 1)], np.int32)
    prediction = -1 if votes[0] == votes[1] else int(np.argmax(votes))
    return dict(step_color=colors, step_identified=identified, votes=votes,
                prediction=np.asarray(prediction, np.int32))


def _color(items, position):
    if not np.all((position >= 0) & (position < items.shape[-1])):
        return -1
    value = items[:, position[0], position[1]]
    return int(np.argmax(value)) if value.any() else -1


def track_movie(frames, radius=2):
    """Infer one public contact per consecutive pair of visible teacher positions."""
    items, occupancy = _pixels(frames, radius)
    side = items.shape[-1]
    colors = np.full(len(items) - 1, -1, np.int8)
    identified = np.zeros(len(colors), bool)
    visible = occupancy.any((1, 2))
    for index in range(len(colors)):
        if not (visible[index] and visible[index + 1]):
            continue
        current = np.argwhere(occupancy[index])[0]
        following = np.argwhere(occupancy[index + 1])[0]
        move = following - radius + SOUTH
        destination = current + move
        if not np.any(np.all(MOVES == move, axis=1)) or np.any((destination < 0) | (destination >= side)):
            continue
        identified[index] = True
        colors[index] = _color(items[index], destination)
    return _finish(colors, identified)


def image_shifts(items):
    """Choose a unique minimum mismatch fraction across overlapping A/B pixels."""
    side = items.shape[-1]
    shifts = np.zeros((len(items) - 1, 2), np.int32)
    numerators = np.empty((len(shifts), len(MOVES)), np.int64)
    denominators = np.empty(len(MOVES), np.int64)
    for candidate, (row, column) in enumerate(MOVES):
        old = items[:-1, :, max(row, 0):side + min(row, 0),
                    max(column, 0):side + min(column, 0)]
        new = items[1:, :, max(-row, 0):side - max(row, 0),
                    max(-column, 0):side - max(column, 0)]
        numerators[:, candidate] = np.count_nonzero(old != new, axis=(1, 2, 3))
        denominators[candidate] = int(np.prod(old.shape[1:]))
    # Compare each candidate to every alternative using exact integer ratios.
    minima = (numerators[:, :, None] * denominators[None, None, :]
              <= numerators[:, None, :] * denominators[None, :, None]).all(axis=2)
    identified = minima.sum(axis=1) == 1
    shifts[identified] = MOVES[np.argmax(minima[identified], axis=1)]
    return shifts, identified


def scene_movie(frames, radius=2):
    """Infer contacts from two camera shifts using item images alone."""
    items, _ = _pixels(frames, radius)
    shifts, shift_identified = image_shifts(items)
    colors = np.full(len(items) - 2, -1, np.int8)
    identified = np.zeros(len(colors), bool)
    side = items.shape[-1]
    for index in range(len(colors)):
        if not shift_identified[index:index + 2].all():
            continue
        destination = radius + shifts[index] + shifts[index + 1] - SOUTH
        if np.any((destination < 0) | (destination >= side)):
            continue
        identified[index] = True
        colors[index] = _color(items[index], destination)
    return dict(**_finish(colors, identified), shifts=shifts, shift_identified=shift_identified)


def decode_frames(frames, rule, radius=2):
    """Apply labels only when scoring already computed predictions."""
    if isinstance(rule, bool) or not isinstance(rule, Integral) or rule not in (0, 1):
        raise ValueError("rule must be zero or one")
    result = {}
    for method, decoder in (("track", track_movie), ("scene", scene_movie)):
        decoded = decoder(frames, radius)
        prediction = int(decoded["prediction"])
        decoded["score"] = np.asarray(.5 if prediction < 0 else float(prediction == rule), np.float64)
        result[method] = decoded
    return result


def family_keys(master_seed, index):
    from scripts import social_learning
    return social_learning.family_keys(master_seed, index)[:2].copy()


def evaluate_family(flat, keys, index, c=None):
    from scripts import social_learning
    c = social_learning.CFG if c is None else c
    social_learning._config(c)
    keys = np.asarray(keys)
    if (keys.shape != (2, 2) or keys.dtype != np.uint32
            or len({key.tobytes() for key in keys}) != 2
            or isinstance(index, bool) or not isinstance(index, Integral) or index < 0):
        raise ValueError("two distinct teacher keys and a nonnegative family index are required")
    rule, swap = int(index) % 2, (int(index) // 2) % 2
    row = dict(keys=keys.copy(), family_index=np.asarray(index, np.int32), rule=np.asarray(rule, np.int32),
               yoked_swap=np.asarray(swap, bool), steps=np.asarray(c.T, np.int32),
               watch_start=np.asarray(c.T // 2, np.int32), radius=np.asarray(c.R, np.int32),
               grid_size=np.asarray(c.G, np.int32), initial_energy=np.asarray(c.e0, np.float32))
    row.update(social_learning._record_teacher(flat, keys, rule, c))
    movie, row["camera_positions"] = social_learning.make_movie(row, c)
    for arm, frames in social_learning.movie_conditions(movie, swap, c).items():
        row[arm + "_frames"] = frames
        for method, decoded in decode_frames(frames, rule, c.R).items():
            row.update({method + "_" + arm + "_" + name: value for name, value in decoded.items()})
    validate_family(row, c)
    return row


def validate_decoding(row):
    rule, radius = int(row["rule"]), int(row["radius"])
    for arm in ARMS:
        for method, decoded in decode_frames(row[arm + "_frames"], rule, radius).items():
            for name, expected in decoded.items():
                actual = np.asarray(row[method + "_" + arm + "_" + name])
                if actual.shape != expected.shape or not np.array_equal(actual, expected):
                    raise ValueError("saved decoder evidence disagrees with public movie pixels")


def validate_family(row, c):
    from scripts import social_learning
    if (int(row["steps"]) != c.T or int(row["watch_start"]) != c.T // 2
            or int(row["radius"]) != c.R or int(row["grid_size"]) != c.G
            or float(row["initial_energy"]) != c.e0):
        raise ValueError("channel family configuration differs from the protocol")
    index = int(row["family_index"])
    if int(row["rule"]) != index % 2 or bool(row["yoked_swap"]) != bool((index // 2) % 2):
        raise ValueError("channel rule or yoked assignment differs from the family")
    social_learning._events(row, "teacher_", c.T)
    movie, camera = social_learning.make_movie(row, c)
    if not np.array_equal(camera, row["camera_positions"]):
        raise ValueError("camera evidence differs from the registered external viewpoint")
    for arm, expected in social_learning.movie_conditions(movie, bool(row["yoked_swap"]), c).items():
        if not np.array_equal(row[arm + "_frames"], expected):
            raise ValueError("movie differs from its recorded teacher and control transformation")
    validate_decoding(row)


def _stats(values):
    values = np.asarray(values, np.float64)
    return dict(mean=float(values.mean()),
                se=float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None)


def summarize(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("at least one completed family is required")
    for row in rows:
        validate_decoding(row)
    result = dict(families=len(rows), rule_counts={str(rule): sum(int(row["rule"]) == rule for row in rows)
                                                for rule in (0, 1)})
    scores = {}
    for method in METHODS:
        estimates = {}
        for arm in ARMS:
            prefix = method + "_" + arm + "_"
            scores[method, arm] = np.asarray([float(row[prefix + "score"]) for row in rows])
            votes = np.array([np.asarray(row[prefix + "votes"]).sum() for row in rows])
            tied = np.array([int(row[prefix + "prediction"]) == -1 for row in rows])
            estimates[arm] = dict(accuracy=_stats(scores[method, arm]), coverage=_stats(votes > 0),
                no_vote_fraction=_stats(votes == 0), tie_fraction=_stats(tied),
                votes=_stats(votes), identifiable_steps=_stats([
                    np.count_nonzero(row[prefix + "step_identified"]) for row in rows]))
        result[method] = dict(arms=estimates, chance_difference=_stats(scores[method, "observed"] - .5),
            comparisons={arm: _stats(scores[method, "observed"] - scores[method, arm]) for arm in ARMS[1:]})
    observed = result["track"]["arms"]["observed"]
    chance, yoked = result["track"]["chance_difference"], result["track"]["comparisons"]["yoked"]
    defined = chance["se"] is not None and yoked["se"] is not None
    gate = dict(coverage_passed=observed["coverage"]["mean"] >= .5,
                accuracy_passed=observed["accuracy"]["mean"] >= .6,
                chance_passed=chance["mean"] > 2 * chance["se"] if defined else None,
                yoked_passed=yoked["mean"] > 2 * yoked["se"] if defined else None)
    result["gate"] = dict(passed=all(gate.values()) if defined else None, **gate)
    return result
