"""Conditional E122 branch O: select observers of a fixed public teacher."""
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from scripts import cold_donor, shelter_evolution, social_learning as social
from seed import seed


PROTOCOL = "social-observer-train-v1"
CFG = social.CFG
ARMS = social.ARMS + ("starting_observed",)
CONTROLS = social.CONTROLS + ("starting_observed",)
family_keys = social.family_keys


def _genome(flat, c):
    flat = np.asarray(flat)
    if flat.shape != (seed.genome_size(c),) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("observer and teacher genomes must be finite float32 rules vectors")
    return flat


def average_centered_ranks(values):
    """Equal scores receive their mean zero-based rank, centered at zero."""
    values = jnp.asarray(values)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("ranking requires at least two candidate scores")
    lower = (values[None, :] < values[:, None]).sum(1)
    equal = (values[None, :] == values[:, None]).sum(1)
    return (lower + .5 * (equal - 1)) / (values.size - 1) - .5


def es_step(state, evaluate, pop, E, sigma, lr, l2=.005):
    """One attempted round, preserving all optimizer state when no scores differ."""
    if (isinstance(pop, bool) or not isinstance(pop, Integral) or pop < 2 or pop % 2
            or isinstance(E, bool) or not isinstance(E, Integral) or E < 2 or E % 2
            or not np.isfinite([sigma, lr, l2]).all() or sigma <= 0 or lr <= 0 or l2 < 0):
        raise ValueError("positive even populations/worlds and finite optimizer settings are required")
    mu, m, v = state["mu"], state["m"], state["v"]
    key, noise_key, world_key, birth_key = jax.random.split(state["key"], 4)
    noise = jax.random.normal(noise_key, (pop // 2, mu.size))
    noise = jnp.concatenate((noise, -noise))
    candidates = mu + sigma * noise
    world_keys = jax.random.split(world_key, E)
    birth_keys = jax.random.split(birth_key, pop // 2 * E).reshape(pop // 2, E, 2)
    birth_keys = jnp.concatenate((birth_keys, birth_keys))
    result = evaluate(candidates, world_keys, birth_keys)
    scores = jnp.asarray(result["fitness"])
    if scores.shape != (pop, E) or not np.isfinite(np.asarray(scores)).all():
        raise ValueError("candidate fitness must be finite with one value per assigned family")
    fitness = scores.mean(1)
    ranks = average_centered_ranks(fitness)
    informative = bool(np.any(np.asarray(fitness) != float(fitness[0])))
    updates = int(state["t"])
    if informative:
        gradient = (ranks[:, None] * noise).sum(0) / (pop * sigma) - l2 * mu
        updates += 1
        m = .9 * m + .1 * gradient
        v = .999 * v + .001 * gradient ** 2
        mu = mu + lr * (m / (1 - .9 ** updates)) / (jnp.sqrt(v / (1 - .999 ** updates)) + 1e-8)
    counts = np.asarray(result["counts"]).sum((0, 1))
    accuracy = counts[:, 0] / np.maximum(counts.sum(1), 1)
    row = dict(informative=informative, all_tied=not informative, t=updates,
               optimizer_updates=updates, distinct_scores=int(np.unique(np.asarray(fitness)).size),
               fit_mean=float(fitness.mean()), fit_best=float(fitness.max()), fit_std=float(fitness.std()),
               acc_first=float(accuracy[0]), acc_second=float(accuracy[1]),
               survived=float(np.asarray(result["survived"]).mean()))
    return dict(mu=mu, m=m, v=v, t=updates, key=key), row


def make_training_batch(teacher_flat, world_keys, c=CFG):
    """Produce each fixed teacher movie once, without a candidate dimension."""
    social._config(c)
    teacher_flat = _genome(teacher_flat, c)
    world_keys = np.asarray(world_keys)
    if (world_keys.ndim != 2 or world_keys.shape[1] != 2 or world_keys.dtype != np.uint32
            or len(world_keys) < 2 or len(world_keys) % 2
            or len({key.tobytes() for key in world_keys}) != len(world_keys)):
        raise ValueError("training requires distinct balanced-family uint32 root keys")
    role_keys = np.stack([np.asarray(jax.random.split(key, 3)) for key in world_keys])
    if len({key.tobytes() for key in role_keys.reshape(-1, 2)}) != role_keys.size // 2:
        raise ValueError("teacher and solo roles must have distinct keys")
    raw, movies, cameras = [], [], []
    for index, keys in enumerate(role_keys):
        teacher = social._record_teacher(teacher_flat, keys[:2], index % 2, c)
        movie, camera = social.make_movie(teacher, c)
        raw.append(teacher)
        movies.append(movie)
        cameras.append(camera)
    result = dict(teacher_genome=teacher_flat.copy(), world_keys=world_keys.copy(), role_keys=role_keys,
                  solo_world_keys=role_keys[:, 2].copy(), rules=np.arange(len(raw), dtype=np.int32) % 2,
                  movies=np.stack(movies), camera_positions=np.stack(cameras))
    result.update({name: np.stack([row[name] for row in raw]) for name in raw[0]})
    return result


def make_evaluator(c=CFG):
    """Compile once; new movie values and genomes remain dynamic inputs."""
    social._config(c)
    unravel = shelter_evolution._unravel(c)

    def life(flat, movie, world_key, birth_key, rule):
        learned = social.watch(flat, birth_key, movie, c)["final"]
        solo = seed.rollout(unravel(flat), world_key, birth_key, rule, c, plastic=True, diagnostics=True,
                            _initial_weights=learned["W"][None],
                            _initial_brain=dict(x=learned["x"][None], Wp=learned["Wp"][None]))
        events = solo["events"]
        encountered = (events["food"] | events["poison"])[:, 0]
        first = jnp.argmax(encountered)
        first_food = jnp.where(encountered.any(), events["food"][first, 0], False)
        return dict(fitness=first_food.astype(jnp.float32), counts=solo["counts"], survived=solo["survived"],
                    **{"candidate_" + name: events[name].astype(bool) for name in social.EVENTS})

    def candidate(flat, movies, world_keys, birth_keys, rules):
        return jax.vmap(life, in_axes=(None, 0, 0, 0, 0))(flat, movies, world_keys, birth_keys, rules)

    @jax.jit
    def evaluate(population, movies, solo_world_keys, birth_keys, rules):
        return jax.vmap(candidate, in_axes=(0, None, None, 0, None))(
            population, movies, solo_world_keys, birth_keys, rules)

    return evaluate


def evaluate_family(teacher_flat, starting_flat, evolved_flat, keys, index, c=CFG):
    teacher_flat, starting_flat, evolved_flat = [_genome(value, c) for value in
                                                 (teacher_flat, starting_flat, evolved_flat)]
    if not np.array_equal(teacher_flat, starting_flat):
        raise ValueError("the fixed teacher and original observer must use the registered original genome")
    row = social.evaluate_family(evolved_flat, keys, index, c, teacher_flat=teacher_flat)
    row.update(teacher_genome=teacher_flat.copy(), starting_genome=starting_flat.copy(),
               evolved_genome=evolved_flat.copy())
    original = social.record_pupil(starting_flat, keys[2], keys[3], index % 2, row["observed_frames"], c)
    row.update({"starting_observed_" + name: value for name, value in original.items()})
    validate_family(row, c)
    return row


def validate_family(row, c=CFG):
    social.validate_family(row, c)
    for key in ("teacher_genome", "starting_genome", "evolved_genome"):
        _genome(row[key], c)
    if not np.array_equal(row["teacher_genome"], row["starting_genome"]):
        raise ValueError("teacher identity changed with the evolved observer")
    prefix = "starting_observed_"
    if not np.array_equal(row[prefix + "frames"], row["observed_frames"]):
        raise ValueError("starting observer must see the identical informative movie")
    _, inputs, outputs, neurons, _ = seed.dims(c)
    motors = cold_donor._array(row, prefix + "watch_motor", (c.T // 2, outputs - 1))
    stay = np.zeros_like(motors)
    stay[:, 0] = 1
    if not np.array_equal(motors, stay) or np.any(row[prefix + "watch_denergy"] != 0):
        raise ValueError("starting observer must remain passive with zero viewing reward")
    cold_donor._array(row, prefix + "watch_denergy", (c.T // 2,))
    for name in ("W", "x", "Wp"):
        for stage in ("watch_initial", "watch_final", "solo_initial"):
            hashes = cold_donor._array(row, prefix + stage + "_" + name + "_sha256", (32,), integer=True)
            if hashes.dtype != np.uint8:
                raise ValueError("starting neural-state fingerprints must be SHA256 bytes")
        if not np.array_equal(row[prefix + "watch_final_" + name + "_sha256"],
                              row[prefix + "solo_initial_" + name + "_sha256"]):
            raise ValueError("starting observer must carry its own acquired neural state")
    if (not np.array_equal(row[prefix + "watch_initial_x_sha256"], social._hash(np.zeros(neurons, np.float32)))
            or not np.array_equal(row[prefix + "watch_initial_Wp_sha256"], social._hash(np.zeros((inputs, c.H), np.float32)))
            or not np.array_equal(row[prefix + "birth_sha256"], row[prefix + "watch_initial_W_sha256"])):
        raise ValueError("starting observer must be a newborn")
    for name, value in (("energy", c.e0), ("alive", 1.), ("denergy", 0.)):
        if not np.array_equal(cold_donor._array(row, prefix + "solo_initial_" + name, (1,)),
                              np.array([value], np.float32)):
            raise ValueError("starting observer body must reset before its solo life")
    food, poison, _, _ = social._events(row, prefix, c.T)
    if float(row[prefix + "fitness"]) != float(food.sum() - poison.sum()):
        raise ValueError("starting food return differs from events")
    for name, value in social.first_meal(food, poison).items():
        if not np.array_equal(row[prefix + name], value):
            raise ValueError("starting first meal differs from events")
    if cold_donor._scalar(row, prefix + "weight_change_count", integer=True) > neurons * neurons:
        raise ValueError("starting weight-change count exceeds its brain")


def summarize(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("at least one completed family is required")
    c = CFG._replace(T=int(rows[0]["steps"]), G=int(rows[0]["grid_size"]), R=int(rows[0]["radius"]),
                     H=int(rows[0]["hidden_units"]), e0=float(rows[0]["initial_energy"]))
    for row in rows:
        validate_family(row, c)
    result = social.summarize(rows)
    first = np.asarray([bool(row["starting_observed_first_food"]) for row in rows])
    survival = np.asarray([bool(row["starting_observed_alive_after"][-1, 0]) for row in rows])
    result["arms"]["starting_observed"] = dict(
        first_food=cold_donor._mean_se(first), survival=cold_donor._mean_se(survival),
        fitness=cold_donor._mean_se([float(row["starting_observed_fitness"]) for row in rows]),
        first_food_count=int(first.sum()),
        first_poison_count=sum(bool(row["starting_observed_first_poison"]) for row in rows),
        no_meal_count=sum(bool(row["starting_observed_no_meal"]) for row in rows),
        survivors=int(survival.sum()),
        changed_weights=cold_donor._mean_se([int(row["starting_observed_weight_change_count"]) for row in rows]))
    food_diff = cold_donor._mean_se(np.asarray([bool(row["observed_first_food"]) for row in rows], float) - first)
    survival_diff = cold_donor._mean_se(np.asarray([bool(row["observed_alive_after"][-1, 0]) for row in rows], float) - survival)
    result["contrasts"]["starting_observed"] = dict(first_food_difference=food_diff,
        survival_difference=survival_diff, gate=social._gate(food_diff, survival_diff))
    defined = len(rows) > 1
    first_passed = all(item["gate"]["first_food_passed"] for item in result["contrasts"].values()) if defined else None
    survival_passed = all(item["gate"]["survival_passed"] for item in result["contrasts"].values()) if defined else None
    result["gate"] = dict(passed=first_passed and survival_passed if defined else None,
                          first_food_passed=first_passed, survival_passed=survival_passed)
    return result
