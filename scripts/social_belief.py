"""Conditional E123: calibrate public contact evidence, then freeze its action readout."""
import hashlib
import json
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from scripts import cold_donor, social_channel, social_learning as social
from seed import seed


PROTOCOL = "social-belief-readout-v1"
CFG = social.CFG
ARMS = ("observed", "yoked", "scene_only", "no_demo", "belief_reset", "birth_parent")
CONTROLS = ARMS[1:]
LAMBDA_GRID = (-4., -2., -1., -.5, 0., .5, 1., 2., 4.)
KEY_ORDER = social.KEY_ORDER
family_keys = social.family_keys
FIT_FIELDS = frozenset(("z", "votes", "contact_color", "contact_outcome"))


def belief_probability(z, lambda_value):
    if not np.isfinite(z) or abs(z) > 1 or lambda_value not in LAMBDA_GRID:
        raise ValueError("belief requires bounded public evidence and a registered lambda")
    return float(1. / (1. + np.exp(-float(lambda_value) * float(z))))


def decode_movie(frames, lambda_value, c=CFG):
    """This inference boundary accepts public pixels, never labels or teacher state."""
    decoded = social_channel.track_movie(frames, c.R)
    votes = decoded["votes"]
    z = float(int(votes[0]) - int(votes[1])) / max(1, int(votes.sum()))
    return {**{"track_" + name: value for name, value in decoded.items()},
            "z": np.asarray(z, np.float64),
            "q": np.asarray(belief_probability(z, lambda_value), np.float32)}


def _base(flat, keys, index, c):
    social._config(c)
    flat, keys = np.asarray(flat), np.asarray(keys)
    if (c.T < 6 or flat.shape != (seed.genome_size(c),) or flat.dtype != np.float32
            or not np.isfinite(flat).all() or keys.shape != (4, 2) or keys.dtype != np.uint32
            or len({key.tobytes() for key in keys}) != 4
            or not isinstance(index, Integral) or isinstance(index, (bool, np.bool_)) or index < 0):
        raise ValueError("provide a valid original genome, four distinct keys and family index")
    rule, swap = int(index) % 2, (int(index) // 2) % 2
    row = dict(keys=keys.copy(), rule=np.asarray(rule, np.int32), yoked_swap=np.asarray(swap, bool),
               family_index=np.asarray(index, np.int32), steps=np.asarray(c.T, np.int32),
               watch_start=np.asarray(c.T // 2, np.int32), agents=np.asarray(1, np.int32),
               radius=np.asarray(c.R, np.int32), grid_size=np.asarray(c.G, np.int32),
               hidden_units=np.asarray(c.H, np.int32), initial_energy=np.asarray(c.e0, np.float32))
    row.update(social._record_teacher(flat, keys, rule, c))
    movie, row["camera_positions"] = social.make_movie(row, c)
    grid, position = seed.init_world(keys[2], c)
    row.update(solo_initial_grid=np.asarray(grid, np.uint8), solo_initial_position=np.asarray(position, np.int32))
    return row, movie


def _birth_observer(flat, pupil_key, c):
    # Reuse the compiled observation birth with zero neural ticks. Eager develop
    # rounds some float32 weights differently from watch and normal rollout.
    empty = np.zeros((0, seed.dims(c)[1]), np.float32)
    return social.watch(flat, pupil_key, empty, c)


def first_contact(initial_grid, grids, positions, energies, alive_before, c=CFG):
    """Infer a contact only from the pupil's local arrival and its own energy change."""
    previous_grid = np.concatenate((np.asarray(initial_grid)[None], np.asarray(grids)[:-1]))
    positions = np.asarray(positions)
    previous_energy = np.concatenate((np.full((1, 1), c.e0, np.float32), np.asarray(energies)[:-1]))
    delta = (np.asarray(energies) - previous_energy) * np.asarray(alive_before)
    colors = previous_grid[np.arange(c.T), positions[:, 0, 0], positions[:, 0, 1]]
    contact = (colors > 0) & np.asarray(alive_before)[:, 0].astype(bool)
    if not contact.any():
        return dict(contact_color=np.asarray(-1, np.int32), contact_outcome=np.asarray(0, np.int32),
                    contact_step=np.asarray(c.T + 1, np.int32), contact_denergy=np.asarray(0., np.float32))
    index = int(contact.argmax())
    own_delta = float(delta[index, 0])
    if not np.isfinite(own_delta) or own_delta == 0:
        raise ValueError("a first contacted item must have an unambiguous own energy outcome")
    return dict(contact_color=np.asarray(colors[index] - 1, np.int32),
                contact_outcome=np.asarray(1 if own_delta > 0 else -1, np.int32),
                contact_step=np.asarray(index + 1, np.int32), contact_denergy=np.asarray(own_delta, np.float32))


def calibration_family(flat, keys, index, c=CFG):
    row, movie = _base(flat, keys, index, c)
    row["observed_frames"] = movie
    row.update(decode_movie(movie, 0., c))
    birth = _birth_observer(flat, keys[3], c)
    empty = np.zeros((0, seed.dims(c)[1]), np.float32)
    pupil = social.record_pupil(flat, keys[2], keys[3], int(row["rule"]), empty, c,
                                _observer=birth, _record_solo=True)
    # Prefix the usual E120 evidence; the unassisted pupil has no watch steps.
    row.update({"pupil_" + name: value for name, value in pupil.items()})
    for name in ("grid", "position", "energy"):
        row["pupil_" + name] = row.pop("pupil_solo_" + name)
    row.update(first_contact(row["solo_initial_grid"], row["pupil_grid"], row["pupil_position"],
                             row["pupil_energy"], row["pupil_alive_before"], c))
    validate_calibration_family(row, c)
    return row


def calibration_input(row):
    """Whitelist the four public/own-outcome fields allowed through the fitter boundary."""
    return dict(z=float(row["z"]), votes=np.asarray(row["track_votes"]).copy(),
                contact_color=int(row["contact_color"]), contact_outcome=int(row["contact_outcome"]))


def calibrate(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("calibration requires assigned families, including any missing contacts")
    for row in rows:
        if set(row) != FIT_FIELDS:
            raise ValueError("the fitter accepts only public evidence and own contact outcome")
        votes = np.asarray(row["votes"])
        color, outcome = row["contact_color"], row["contact_outcome"]
        if (votes.shape != (2,) or votes.dtype.kind not in "iu" or np.any(votes < 0)
                or color not in (-1, 0, 1) or outcome not in (-1, 0, 1)
                or (color == -1) != (outcome == 0)
                or not np.isfinite(row["z"])
                or float(row["z"]) != float(int(votes[0]) - int(votes[1])) / max(1, int(votes.sum()))):
            raise ValueError("invalid public vote or own-outcome calibration input")
    contacts = [row for row in rows if row["contact_color"] != -1]
    losses = []
    for lambda_value in LAMBDA_GRID:
        # A-contact logit is lambda*z; a B-contact inverts that same prediction.
        logits = [lambda_value * row["z"] * (1 if row["contact_color"] == 0 else -1) for row in contacts]
        log_losses = [float(np.logaddexp(0., logit) - (row["contact_outcome"] > 0) * logit)
                      for row, logit in zip(contacts, logits)]
        loss_stats = cold_donor._mean_se(log_losses) if contacts else dict(mean=None, se=None)
        losses.append({"lambda": lambda_value, "mean_log_loss": loss_stats["mean"],
                       "se_log_loss": loss_stats["se"]})
    chosen = min(losses, key=lambda row: (row["mean_log_loss"], abs(row["lambda"]),
                                          -row["lambda"]))["lambda"] if contacts else 0.
    payload = [{**row, "votes": np.asarray(row["votes"]).tolist()} for row in rows]
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(selected_lambda=chosen, families=len(rows), contacts=len(contacts), losses=losses,
                contact_coverage=cold_donor._mean_se([row["contact_color"] != -1 for row in rows]),
                vote_coverage=cold_donor._mean_se([np.asarray(row["votes"]).sum() > 0 for row in rows]),
                vote_tie_fraction=cold_donor._mean_se([row["votes"][0] == row["votes"][1] for row in rows]),
                calibration_inputs_sha256=fingerprint)


def evaluate_family(flat, keys, index, lambda_value, c=CFG):
    if lambda_value not in LAMBDA_GRID:
        raise ValueError("evaluation requires the frozen calibrated lambda")
    row, movie = _base(flat, keys, index, c)
    row["lambda_value"] = np.asarray(lambda_value, np.float64)
    conditions = social.movie_conditions(movie, bool(row["yoked_swap"]), c)
    observed_state = None
    for arm in ARMS:
        if arm == "birth_parent":
            frames = np.zeros((0, seed.dims(c)[1]), np.float32)
            observer = _birth_observer(flat, keys[3], c)
            decoded = dict(track_step_color=np.zeros(0, np.int8), track_step_identified=np.zeros(0, bool),
                           track_votes=np.zeros(2, np.int32), track_prediction=np.asarray(-1, np.int32),
                           z=np.asarray(0., np.float64), q=np.asarray(.5, np.float32))
        else:
            frames = movie if arm == "belief_reset" else conditions[arm]
            observer = observed_state if arm == "belief_reset" else social.watch(flat, keys[3], frames, c)
            decoded = decode_movie(frames, lambda_value, c)
            if arm == "observed":
                observed_state = observer
            if arm == "belief_reset":
                decoded["q"] = np.asarray(.5, np.float32)
        assisted = arm != "birth_parent"
        recorded = social.record_pupil(flat, keys[2], keys[3], int(row["rule"]), frames, c,
                                       _observer=observer, _record_solo=True,
                                       _belief_q=jnp.asarray([decoded["q"]]) if assisted else None)
        recorded.update(decoded, readout_enabled=np.asarray(assisted, bool))
        row.update({arm + "_" + name: value for name, value in recorded.items()})
    validate_family(row, c)
    return row


def _metadata(row, c):
    for name, expected in (("steps", c.T), ("watch_start", c.T // 2), ("agents", 1),
                           ("radius", c.R), ("grid_size", c.G), ("hidden_units", c.H)):
        if cold_donor._scalar(row, name, integer=True) != expected:
            raise ValueError("family configuration differs from its protocol")
    index = cold_donor._scalar(row, "family_index", integer=True)
    if (int(row["rule"]) != index % 2 or bool(row["yoked_swap"]) != bool((index // 2) % 2)
            or float(row["initial_energy"]) != np.float32(c.e0)):
        raise ValueError("family rule assignment or body configuration differs")
    keys = cold_donor._array(row, "keys", (4, 2), integer=True)
    if keys.dtype != np.uint32 or len({key.tobytes() for key in keys}) != 4:
        raise ValueError("family random keys are invalid or reused")
    social._events(row, "teacher_", c.T)
    movie, camera = social.make_movie(row, c)
    if not np.array_equal(camera, row["camera_positions"]):
        raise ValueError("teacher movie camera differs from the recorded life")
    return movie


def _validate_neural(row, prefix, c, *, birth=False):
    _, inputs, _, neurons, _ = seed.dims(c)
    for name in ("W", "x", "Wp"):
        for stage in ("watch_initial", "watch_final", "solo_initial"):
            value = cold_donor._array(row, prefix + stage + "_" + name + "_sha256", (32,), integer=True)
            if value.dtype != np.uint8:
                raise ValueError("neural state hashes must be SHA256 bytes")
        if not np.array_equal(row[prefix + "watch_final_" + name + "_sha256"],
                              row[prefix + "solo_initial_" + name + "_sha256"]):
            raise ValueError("solo brain differs from its assigned observer state")
        if birth and not np.array_equal(row[prefix + "watch_initial_" + name + "_sha256"],
                                        row[prefix + "watch_final_" + name + "_sha256"]):
            raise ValueError("an unexposed parent cannot acquire a watch state")
    if (not np.array_equal(row[prefix + "birth_sha256"], row[prefix + "watch_initial_W_sha256"])
            or not np.array_equal(row[prefix + "watch_initial_x_sha256"], social._hash(np.zeros(neurons, np.float32)))
            or not np.array_equal(row[prefix + "watch_initial_Wp_sha256"], social._hash(np.zeros((inputs, c.H), np.float32)))):
        raise ValueError("initial observer state differs from its own birth")
    watch_steps = 0 if birth else c.T // 2
    frames = cold_donor._array(row, prefix + "frames", (watch_steps, inputs))
    motors = cold_donor._array(row, prefix + "watch_motor", (watch_steps, 5))
    expected = np.zeros_like(motors)
    expected[:, 0] = 1
    if (not np.array_equal(motors, expected)
            or np.any(cold_donor._array(row, prefix + "watch_denergy", (watch_steps,)) != 0)):
        raise ValueError("passive observer cannot receive another agent's action or reward")
    for name, value in (("energy", c.e0), ("alive", 1.), ("denergy", 0.)):
        if not np.array_equal(row[prefix + "solo_initial_" + name], np.asarray([value], np.float32)):
            raise ValueError("solo body must start fresh")
    food, poison, _, _ = social._events(row, prefix, c.T)
    for name, value in social.first_meal(food, poison).items():
        if not np.array_equal(row[prefix + name], value):
            raise ValueError("first-meal classification differs from actual events")
    if float(row[prefix + "fitness"]) != float(food.sum() - poison.sum()):
        raise ValueError("fitness differs from the actual contact events")
    count = cold_donor._scalar(row, prefix + "weight_change_count", integer=True)
    if count > neurons * neurons or (birth and count != 0):
        raise ValueError("invalid number of changed watch weights")
    return frames


def _trajectory(row, prefix, c, *, calibration=False):
    stem = prefix if calibration else prefix + "solo_"
    grid = cold_donor._array(row, stem + "grid", (c.T, c.G, c.G), integer=True)
    pos = cold_donor._array(row, stem + "position", (c.T, 1, 2), integer=True)
    energy = cold_donor._array(row, stem + "energy", (c.T, 1))
    if (not np.isin(grid, (0, 1, 2)).all() or np.any(pos < 0) or np.any(pos >= c.G)
            or np.any(energy < 0) or np.any(energy > c.e_max)):
        raise ValueError("solo trajectory leaves world or energy bounds")
    first = first_contact(row["solo_initial_grid"], grid, pos, energy, row[prefix + "alive_before"], c)
    if int(first["contact_step"]) != int(row[prefix + "first_meal_step"]):
        raise ValueError("contact timing differs between own visible arrival and encounter events")
    return grid, pos, energy, first


def validate_calibration_family(row, c=CFG):
    movie = _metadata(row, c)
    if not np.array_equal(movie, row["observed_frames"]):
        raise ValueError("calibration movie differs from public teacher evidence")
    for name, expected in decode_movie(movie, 0., c).items():
        if not np.array_equal(row[name], expected):
            raise ValueError("calibration public decoding differs")
    _validate_neural(row, "pupil_", c, birth=True)
    *_, first = _trajectory(row, "pupil_", c, calibration=True)
    for name, expected in first.items():
        if not np.array_equal(row[name], expected):
            raise ValueError("calibration contact differs from pupil's own experience")
    calibrate([calibration_input(row)])


def _validate_readout(row, prefix, grid, pos, energy, c):
    q = float(row[prefix + "q"])
    before_grid = np.concatenate((row["solo_initial_grid"][None], grid[:-1]))
    before_pos = np.concatenate((row["solo_initial_position"][None], pos[:-1]))[:, 0]
    destinations = (before_pos[:, None, :] + social_channel.MOVES[None]) % c.G
    items = before_grid[np.arange(c.T)[:, None], destinations[..., 0], destinations[..., 1]]
    contrast = (items == 1).astype(np.float32) - (items == 2).astype(np.float32)
    expected_bias = np.float32(3.) * (np.float32(2.) * np.float32(q) - np.float32(1.)) * contrast
    bias = cold_donor._array(row, prefix + "belief_bias", (c.T, 1, 5))
    if not np.allclose(bias[:, 0], expected_bias, rtol=0, atol=1e-6) or np.any(np.abs(bias) > 3.):
        raise ValueError("belief action bias differs from the five locally visible destinations")
    actions = cold_donor._array(row, prefix + "belief_action", (c.T, 1), integer=True)[:, 0]
    if np.any(actions < 0) or np.any(actions > 4):
        raise ValueError("invalid sampled action")
    if not np.array_equal(row[prefix + "belief_efference"], np.eye(5, dtype=np.float32)[actions, None]):
        raise ValueError("motor feedback must contain the actual sampled action")
    alive = np.asarray(row[prefix + "alive_before"], np.int32)
    expected_pos = (before_pos + social_channel.MOVES[actions] * alive) % c.G
    if not np.array_equal(pos[:, 0], expected_pos):
        raise ValueError("sampled action does not produce the saved solo movement")
    if not np.array_equal(row[prefix + "belief_q"], np.full((c.T, 1), q, np.float32)):
        raise ValueError("calibrated belief must stay fixed throughout solo life")
    previous_energy = np.concatenate((np.full((1, 1), c.e0, np.float32), energy[:-1]))
    if not np.array_equal(row[prefix + "belief_denergy"], (energy - previous_energy) * alive):
        raise ValueError("recorded reward must be the pupil's own energy change")
    base = cold_donor._array(row, prefix + "belief_unbiased_logits", (c.T, 1, 5))
    used = cold_donor._array(row, prefix + "belief_biased_logits", (c.T, 1, 5))
    if not np.allclose(used, base + bias, rtol=0, atol=1e-6):
        raise ValueError("action sampling logits differ from the fixed bounded readout")


def validate_family(row, c=CFG):
    social.validate_family(row, c)  # Existing four-arm movie/body/neural evidence contract.
    lambda_value = float(row["lambda_value"])
    if lambda_value not in LAMBDA_GRID:
        raise ValueError("evaluation lambda is outside the registered calibration grid")
    for arm in ARMS:
        prefix = arm + "_"
        frames = _validate_neural(row, prefix, c, birth=arm == "birth_parent")
        grid, pos, energy, _ = _trajectory(row, prefix, c)
        if not np.array_equal(row[prefix + "birth_sha256"], row["observed_birth_sha256"]):
            raise ValueError("all six branches must share the same original newborn")
        if arm == "birth_parent":
            if (bool(row[prefix + "readout_enabled"]) or float(row[prefix + "q"]) != .5
                    or float(row[prefix + "z"]) != 0 or np.any(row[prefix + "track_votes"])):
                raise ValueError("birth parent must have no demonstration or readout")
            continue
        decoded = decode_movie(frames, lambda_value, c)
        if arm == "belief_reset":
            decoded["q"] = np.asarray(.5, np.float32)
            for name in ("frames", "watch_motor", "watch_denergy", "weight_change_count"):
                if not np.array_equal(row[prefix + name], row["observed_" + name]):
                    raise ValueError("belief reset must preserve the observed learning experience")
            for name in ("W", "x", "Wp"):
                for stage in ("watch_initial", "watch_final", "solo_initial"):
                    suffix = stage + "_" + name + "_sha256"
                    if not np.array_equal(row[prefix + suffix], row["observed_" + suffix]):
                        raise ValueError("belief reset must preserve the exact acquired neural state")
        for name, expected in decoded.items():
            if not np.array_equal(row[prefix + name], expected):
                raise ValueError("belief differs from public movie evidence or its registered reset")
        if not bool(row[prefix + "readout_enabled"]):
            raise ValueError("observer branches must use their assigned belief readout")
        _validate_readout(row, prefix, grid, pos, energy, c)


def summarize(rows):
    rows = list(rows)
    result = social.summarize(rows)
    c = CFG._replace(T=int(rows[0]["steps"]), G=int(rows[0]["grid_size"]), R=int(rows[0]["radius"]),
                     H=int(rows[0]["hidden_units"]), e0=float(rows[0]["initial_energy"]))
    lambda_value = float(rows[0]["lambda_value"])
    for row in rows:
        validate_family(row, c)
        if float(row["lambda_value"]) != lambda_value:
            raise ValueError("one frozen calibrated lambda is required throughout each lineage")
    vectors, arms = {}, {}
    for arm in ARMS:
        first = np.asarray([bool(row[arm + "_first_food"]) for row in rows], float)
        survival = np.asarray([bool(row[arm + "_alive_after"][-1, 0]) for row in rows], float)
        vectors[arm] = (first, survival)
        arms[arm] = dict(first_food=cold_donor._mean_se(first), survival=cold_donor._mean_se(survival),
                         fitness=cold_donor._mean_se([float(row[arm + "_fitness"]) for row in rows]),
                         first_food_count=int(first.sum()), survivors=int(survival.sum()),
                         first_poison_count=sum(bool(row[arm + "_first_poison"]) for row in rows),
                         no_meal_count=sum(bool(row[arm + "_no_meal"]) for row in rows),
                         no_meal=cold_donor._mean_se([bool(row[arm + "_no_meal"]) for row in rows]),
                         first_meal_step=cold_donor._mean_se([int(row[arm + "_first_meal_step"]) for row in rows]),
                         changed_weights=cold_donor._mean_se([int(row[arm + "_weight_change_count"]) for row in rows]),
                         belief_probability=cold_donor._mean_se([float(row[arm + "_q"]) for row in rows]))
    contrasts = {}
    for arm in CONTROLS:
        first = cold_donor._mean_se(vectors["observed"][0] - vectors[arm][0])
        survival = cold_donor._mean_se(vectors["observed"][1] - vectors[arm][1])
        contrasts[arm] = dict(first_food_difference=first, survival_difference=survival,
                              gate=social._gate(first, survival))
    defined = len(rows) > 1
    primary = all(value["gate"]["first_food_passed"] for value in contrasts.values()) if defined else None
    safeguard = all(value["gate"]["survival_passed"] for value in contrasts.values()) if defined else None
    result.update(arms=arms, contrasts=contrasts, selected_lambda=lambda_value,
                  gate=dict(passed=primary and safeguard if defined else None,
                            first_food_passed=primary, survival_passed=safeguard))
    return result
