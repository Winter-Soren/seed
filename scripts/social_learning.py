"""E120 observation of a lived demonstrator followed by a fresh solo lifetime."""
from functools import partial
import hashlib
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from scripts import cold_donor, shelter_evolution
from seed import seed


PROTOCOL = "social-learning-v1"
CFG = seed.Cfg(M=1)
ARMS = ("observed", "yoked", "no_demo", "scene_only")
CONTROLS = ARMS[1:]
EVENTS = ("food", "poison", "alive_before", "alive_after")
KEY_ORDER = ("teacher_world", "teacher_birth", "solo_world", "pupil_birth")


def family_keys(master_seed, index):
    shelter_evolution.evaluation_keys(master_seed, index)
    return np.asarray(jax.random.split(jax.random.fold_in(
        jax.random.PRNGKey(int(master_seed)), int(index)), 4))


def _config(c):
    if c.M != 1 or c.T < 2 or c.T % 2 or c.R < 1:
        raise ValueError("social learning requires one agent, an even life and a visible local neighborhood")


def _hash(value):
    return np.frombuffer(hashlib.sha256(np.asarray(value).tobytes()).digest(), np.uint8)


def make_movie(raw, c=CFG):
    """Reconstruct external pre-action views, with a camera one position behind."""
    _config(c)
    grid = np.concatenate((raw["teacher_initial_grid"][None], raw["teacher_grid"][:-1]))
    position = np.concatenate((raw["teacher_initial_position"][None], raw["teacher_position"][:-1]))[:, 0]
    previous_position = np.concatenate((position[:1], position[:-1]))
    signal = np.concatenate((np.zeros(1, np.float32), raw["teacher_signal"][:-1, 0]))
    alive = np.asarray(raw["teacher_alive_before"], bool)[:, 0]
    camera = (previous_position + np.array([1, 0])) % c.G
    offsets = np.arange(-c.R, c.R + 1)
    frames = []
    for t in range(c.T // 2, c.T):
        rows = (camera[t, 0] + offsets) % c.G
        columns = (camera[t, 1] + offsets) % c.G
        patch = grid[t][rows[:, None], columns[None, :]]
        occupied = ((rows[:, None] == position[t, 0])
                    & (columns[None, :] == position[t, 1]) & alive[t])
        frames.append(np.concatenate(((patch == 1).reshape(-1), (patch == 2).reshape(-1),
                                     occupied.reshape(-1), (occupied * signal[t]).reshape(-1),
                                     np.array([c.e0 - 1., 0., 1.]))).astype(np.float32))
    return np.asarray(frames), camera[c.T // 2:].astype(np.int32)


def movie_conditions(movie, swap, c=CFG):
    area = (2 * c.R + 1) ** 2
    conditions = {name: np.asarray(movie).copy() for name in ARMS}
    if swap:
        conditions["yoked"][:, :area] = movie[:, area:2 * area]
        conditions["yoked"][:, area:2 * area] = movie[:, :area]
    conditions["no_demo"][:, :4 * area] = 0
    conditions["scene_only"][:, 2 * area:4 * area] = 0
    return conditions


@partial(jax.jit, static_argnames=("c",))
def watch(flat, development_key, frames, c=CFG):
    """Passive observer learns only from its own brain and the visible movie."""
    g = shelter_evolution._unravel(c)(flat)
    _, inputs, outputs, neurons, _ = seed.dims(c)
    own_key = jax.random.split(development_key, 1)[0]
    birth, mask = seed.develop(g, own_key, c)
    types = seed.neuron_types(c)
    hebb = g["hebb"][types[:, None], types[None, :]]
    initial = dict(W=birth, x=jnp.zeros(neurons), Wp=jnp.zeros((inputs, c.H)))
    stay = jax.nn.one_hot(0, outputs - 1)

    def step(state, observation):
        weights, activity, predictor, _ = seed.brain_step(
            state["W"], state["x"], state["Wp"], observation, jnp.asarray(0.), mask,
            hebb, jnp.exp(g["log_eta"]), jnp.exp(g["log_plr"]), g["mod"], 1., c)
        activity = activity.at[neurons - outputs:neurons - 1].set(stay)
        return dict(W=weights, x=activity, Wp=predictor), dict(motor=activity[neurons - outputs:neurons - 1],
                                                           denergy=jnp.asarray(0.))

    final, trace = jax.lax.scan(step, initial, frames)
    return dict(initial=initial, final=final, trace=trace, mask=mask)


def _record_teacher(flat, keys, rule, c):
    g = shelter_evolution._unravel(c)(flat)
    result = seed.rollout(g, keys[0], keys[1], rule, c, plastic=True,
                          record=True, diagnostics=True, _return_weights=True)
    result = jax.tree.map(np.asarray, result)
    grid, position = seed.init_world(keys[0], c)
    row = dict(teacher_initial_grid=np.asarray(grid, np.uint8),
               teacher_initial_position=np.asarray(position, np.int32))
    for target, source in (("grid", "grid"), ("position", "pos"), ("signal", "sig"), ("energy", "energy")):
        row["teacher_" + target] = result["traj"][source]
    for event in EVENTS:
        row["teacher_" + event] = np.asarray(result["events"][event], bool)
    row["teacher_birth_sha256"] = _hash(result["_weight_state"]["birth_W"][0])
    row["teacher_terminal_sha256"] = _hash(result["_weight_state"]["terminal_W"][0])
    return row


def first_meal(food, poison):
    food, poison = np.asarray(food, bool), np.asarray(poison, bool)
    encounters = (food | poison)[:, 0]
    exists = bool(encounters.any())
    index = int(encounters.argmax()) if exists else len(encounters)
    return dict(first_food=np.asarray(exists and food[index, 0], bool),
                first_poison=np.asarray(exists and poison[index, 0], bool),
                no_meal=np.asarray(not exists, bool), first_meal_step=np.asarray(index + 1, np.int32))


def record_pupil(flat, solo_key, pupil_key, rule, frames, c=CFG, *,
                 _observer=None, _belief_q=None, _record_solo=False):
    """Record one newborn watching a supplied public movie and then acting alone."""
    row = {}
    g = shelter_evolution._unravel(c)(flat)
    observer = jax.tree.map(np.asarray, watch(flat, pupil_key, frames, c) if _observer is None else _observer)
    initial, final = observer["initial"], observer["final"]
    if any(not np.isfinite(value).all() for value in (*initial.values(), *final.values())):
        raise ValueError("observer neural state must be finite")
    solo = seed.rollout(g, solo_key, pupil_key, rule, c, plastic=True, diagnostics=True,
                        _initial_weights=final["W"][None], _return_weights=True,
                        _initial_brain=dict(x=final["x"][None], Wp=final["Wp"][None]),
                        _belief_q=_belief_q, record=_record_solo)
    solo = jax.tree.map(np.asarray, solo)
    prefix = ""
    row[prefix + "frames"] = frames
    row[prefix + "watch_motor"] = observer["trace"]["motor"]
    row[prefix + "watch_denergy"] = observer["trace"]["denergy"]
    row[prefix + "birth_sha256"] = _hash(initial["W"])
    for name in ("W", "x", "Wp"):
        row[prefix + "watch_initial_" + name + "_sha256"] = _hash(initial[name])
        row[prefix + "watch_final_" + name + "_sha256"] = _hash(final[name])
    actual_initial = dict(W=solo["_weight_state"]["initial_W"][0],
                          x=solo["_initial_brain"]["x"][0], Wp=solo["_initial_brain"]["Wp"][0])
    for name, value in actual_initial.items():
        row[prefix + "solo_initial_" + name + "_sha256"] = _hash(value)
    for name in ("energy", "alive", "denergy"):
        row[prefix + "solo_initial_" + name] = solo["_initial_brain"][name]
    for event in EVENTS:
        row[prefix + event] = np.asarray(solo["events"][event], bool)
    row[prefix + "fitness"] = np.asarray(solo["fitness"])
    row[prefix + "weight_change_count"] = np.asarray(np.count_nonzero(final["W"] != initial["W"]), np.int32)
    row.update({prefix + name: value for name, value in first_meal(
        row[prefix + "food"], row[prefix + "poison"]).items()})
    if _record_solo:
        for name, source in (("grid", "grid"), ("position", "pos"), ("energy", "energy"), ("signal", "sig")):
            row["solo_" + name] = solo["traj"][source]
    if _belief_q is not None:
        row.update({"belief_" + name: value for name, value in solo["_belief"].items()})
    return row


def evaluate_family(flat, keys, index, c=CFG, *, teacher_flat=None):
    _config(c)
    flat, keys = np.asarray(flat), np.asarray(keys)
    if (flat.shape != (seed.genome_size(c),) or flat.dtype != np.float32 or not np.isfinite(flat).all()
            or keys.shape != (4, 2) or keys.dtype != np.uint32
            or len({key.tobytes() for key in keys}) != 4
            or not isinstance(index, Integral) or isinstance(index, (bool, np.bool_)) or index < 0):
        raise ValueError("provide a finite float32 genome, distinct four keys and nonnegative family index")
    rule, swap = int(index) % 2, (int(index) // 2) % 2
    row = dict(keys=keys.copy(), rule=np.asarray(rule, np.int32), yoked_swap=np.asarray(swap, bool),
               family_index=np.asarray(index, np.int32), steps=np.asarray(c.T, np.int32),
               watch_start=np.asarray(c.T // 2, np.int32), agents=np.asarray(1, np.int32),
               radius=np.asarray(c.R, np.int32), grid_size=np.asarray(c.G, np.int32),
               hidden_units=np.asarray(c.H, np.int32),
               initial_energy=np.asarray(c.e0, np.float32))
    row.update(_record_teacher(flat if teacher_flat is None else teacher_flat, keys, rule, c))
    movie, row["camera_positions"] = make_movie(row, c)
    conditions = movie_conditions(movie, swap, c)
    grid, position = seed.init_world(keys[2], c)
    row.update(solo_initial_grid=np.asarray(grid, np.uint8), solo_initial_position=np.asarray(position, np.int32))
    for arm, frames in conditions.items():
        row.update({arm + "_" + name: value for name, value in
                    record_pupil(flat, keys[2], keys[3], rule, frames, c).items()})
    validate_family(row, c)
    return row


def _events(row, prefix, steps):
    arrays = [cold_donor._array(row, prefix + name, (steps, 1), binary=True) for name in EVENTS]
    food, poison, before, after = arrays
    if (not before[0].all() or np.any(food & poison) or np.any((food | poison) & ~before)
            or np.any(after & ~before) or not np.array_equal(before[1:], after[:-1])):
        raise ValueError("encounter or death timing is invalid")
    return arrays


def validate_family(row, c=CFG):
    _config(c)
    for name, value in (("steps", c.T), ("watch_start", c.T // 2), ("agents", 1), ("radius", c.R),
                        ("grid_size", c.G), ("hidden_units", c.H)):
        if cold_donor._scalar(row, name, integer=True) != value:
            raise ValueError("family configuration differs from protocol")
    index = cold_donor._scalar(row, "family_index", integer=True)
    if (cold_donor._scalar(row, "rule", integer=True) != index % 2
            or bool(row["yoked_swap"]) != bool((index // 2) % 2)
            or float(row["initial_energy"]) != np.float32(c.e0)):
        raise ValueError("family rule, yoked assignment or initial energy differs")
    keys = cold_donor._array(row, "keys", (4, 2), integer=True)
    if keys.dtype != np.uint32 or len({key.tobytes() for key in keys}) != 4:
        raise ValueError("family random keys must be distinct uint32 pairs")
    _events(row, "teacher_", c.T)
    for name, shape in (("teacher_initial_grid", (c.G, c.G)), ("teacher_grid", (c.T, c.G, c.G)),
                        ("solo_initial_grid", (c.G, c.G))):
        if not np.isin(cold_donor._array(row, name, shape, integer=True), (0, 1, 2)).all():
            raise ValueError("invalid visible grid")
    for name, shape in (("teacher_initial_position", (1, 2)), ("teacher_position", (c.T, 1, 2)),
                        ("solo_initial_position", (1, 2)), ("camera_positions", (c.T // 2, 2))):
        position = cold_donor._array(row, name, shape, integer=True)
        if np.any(position < 0) or np.any(position >= c.G):
            raise ValueError("position leaves the toroidal world")
    signal = cold_donor._array(row, "teacher_signal", (c.T, 1))
    energy = cold_donor._array(row, "teacher_energy", (c.T, 1))
    if np.any(np.abs(signal) > 1) or np.any(energy < 0) or np.any(energy > c.e_max):
        raise ValueError("teacher signal or energy exceeds engine bounds")
    movie, camera = make_movie(row, c)
    if not np.array_equal(row["camera_positions"], camera):
        raise ValueError("camera differs from the registered previous-position view")
    conditions = movie_conditions(movie, bool(row["yoked_swap"]), c)
    _, inputs, outputs, neurons, _ = seed.dims(c)
    for arm in ARMS:
        prefix = arm + "_"
        frames = cold_donor._array(row, prefix + "frames", (c.T // 2, inputs))
        if not np.array_equal(frames, conditions[arm]):
            raise ValueError("observer frames differ from the registered visible condition")
        motors = cold_donor._array(row, prefix + "watch_motor", (c.T // 2, outputs - 1))
        stay = np.zeros_like(motors)
        stay[:, 0] = 1
        if (not np.array_equal(motors, stay)
                or np.any(cold_donor._array(row, prefix + "watch_denergy", (c.T // 2,)) != 0)):
            raise ValueError("passive observer must stay with zero experienced reward")
        for name in ("W", "x", "Wp"):
            hashes = {}
            for stage in ("watch_initial", "watch_final", "solo_initial"):
                value = cold_donor._array(row, prefix + stage + "_" + name + "_sha256", (32,), integer=True)
                if value.dtype != np.uint8:
                    raise ValueError("neural state hash must contain uint8 SHA256 bytes")
                hashes[stage] = value
            if not np.array_equal(hashes["watch_final"], hashes["solo_initial"]):
                raise ValueError("solo test must carry exactly this observer's learned state")
            if not np.array_equal(hashes["watch_initial"], row["observed_watch_initial_" + name + "_sha256"]):
                raise ValueError("all movie arms must have identical initial brains")
        if (not np.array_equal(row[prefix + "watch_initial_x_sha256"], _hash(np.zeros(neurons, np.float32)))
                or not np.array_equal(row[prefix + "watch_initial_Wp_sha256"], _hash(np.zeros((inputs, c.H), np.float32)))
                or not np.array_equal(row[prefix + "birth_sha256"], row[prefix + "watch_initial_W_sha256"])):
            raise ValueError("observers must begin as independently developed newborns")
        for name, value in (("energy", c.e0), ("alive", 1.), ("denergy", 0.)):
            actual = cold_donor._array(row, prefix + "solo_initial_" + name, (1,))
            if not np.array_equal(actual, np.array([value], np.float32)):
                raise ValueError("solo body must begin fresh after passive observation")
        food, poison, _, _ = _events(row, prefix, c.T)
        for name, value in first_meal(food, poison).items():
            if not np.array_equal(row[prefix + name], value):
                raise ValueError("first-meal classification differs from events")
        if float(row[prefix + "fitness"]) != float(food.sum() - poison.sum()):
            raise ValueError("solo return differs from encounter events")
        if cold_donor._scalar(row, prefix + "weight_change_count", integer=True) > neurons * neurons:
            raise ValueError("changed-weight count exceeds the newborn brain")


def _gate(first_food, survival):
    decided = first_food["se"] is not None and survival["se"] is not None
    primary = bool(first_food["mean"] > 2 * first_food["se"]) if decided else None
    safeguard = bool(survival["mean"] >= -2 * survival["se"]) if decided else None
    return dict(passed=primary and safeguard if decided else None,
                first_food_passed=primary, survival_passed=safeguard)


def summarize(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot summarize an empty social-learning panel")
    steps = int(rows[0]["steps"])
    cfg = CFG._replace(T=steps, G=int(rows[0]["grid_size"]), R=int(rows[0]["radius"]),
                       H=int(rows[0]["hidden_units"]), e0=float(rows[0]["initial_energy"]))
    for row in rows:
        validate_family(row, cfg)
        if (int(row["steps"]) != steps or int(row["agents"]) != 1
                or any(not np.isfinite(np.asarray(value)).all() for value in row.values())):
            raise ValueError("social-learning panel contains inconsistent or nonfinite families")
    seen = set()
    for row in rows:
        for key in row["keys"]:
            if key.tobytes() in seen:
                raise ValueError("random keys were reused across families")
            seen.add(key.tobytes())
    arms, vectors = {}, {}
    for arm in ARMS:
        first = np.asarray([bool(row[arm + "_first_food"]) for row in rows])
        survived = np.asarray([bool(row[arm + "_alive_after"][-1, 0]) for row in rows])
        vectors[arm] = dict(first=first.astype(float), survival=survived.astype(float))
        arms[arm] = dict(first_food=cold_donor._mean_se(first), survival=cold_donor._mean_se(survived),
                         fitness=cold_donor._mean_se([float(row[arm + "_fitness"]) for row in rows]),
                         first_food_count=int(first.sum()),
                         first_poison_count=sum(bool(row[arm + "_first_poison"]) for row in rows),
                         no_meal_count=sum(bool(row[arm + "_no_meal"]) for row in rows),
                         survivors=int(survived.sum()),
                         changed_weights=cold_donor._mean_se([int(row[arm + "_weight_change_count"]) for row in rows]))
    contrasts = {}
    for control in CONTROLS:
        food = cold_donor._mean_se(vectors["observed"]["first"] - vectors[control]["first"])
        survival = cold_donor._mean_se(vectors["observed"]["survival"] - vectors[control]["survival"])
        contrasts[control] = dict(first_food_difference=food, survival_difference=survival, gate=_gate(food, survival))
    defined = len(rows) > 1
    first_passed = all(row["gate"]["first_food_passed"] for row in contrasts.values()) if defined else None
    survival_passed = all(row["gate"]["survival_passed"] for row in contrasts.values()) if defined else None
    teacher = {}
    for phase, bounds in (("practice", (0, steps // 2)), ("demonstration", (steps // 2, steps))):
        food = np.asarray([np.asarray(row["teacher_food"])[slice(*bounds)].sum() for row in rows])
        poison = np.asarray([np.asarray(row["teacher_poison"])[slice(*bounds)].sum() for row in rows])
        teacher[phase] = dict(food=cold_donor._mean_se(food), poison=cold_donor._mean_se(poison),
                              accuracy=cold_donor._ratio_se(food, food + poison),
                              encounters=int((food + poison).sum()))
    area = (2 * cfg.R + 1) ** 2
    teacher.update(survival=cold_donor._mean_se([bool(row["teacher_alive_after"][-1, 0]) for row in rows]),
                   visible_frames=cold_donor._mean_se([np.any(row["observed_frames"][:, 2 * area:3 * area], axis=1).sum()
                                                      for row in rows]))
    return dict(families=len(rows), assigned_newborns=len(rows), arms=arms, contrasts=contrasts, teacher=teacher,
                gate=dict(passed=first_passed and survival_passed if defined else None,
                          first_food_passed=first_passed, survival_passed=survival_passed))
