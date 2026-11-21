"""Conditional E124: three public-behavior handoffs under one frozen controller.

Only the predecessor's actual solo trajectory supplies the next demonstration.
The genome and calibrated lambda never change. Every new cohort develops its own
brain and starts with neutral task belief; no neural state crosses a handoff.
"""
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from scripts import cold_donor, shelter_evolution, social_belief as belief, social_learning as social
from seed import seed


PROTOCOL = "social-transmission-v1"
CFG = social.CFG
COHORTS = 3
ARMS = ("predecessor", "repeated_founder", "yoked", "belief_reset")
CONTROLS = ARMS[1:]
KEY_ORDER = ("solo_world", "pupil_birth")
STAGE_ORDER = ("founder", "c1", "c2", "c3")
SURVIVAL_FLOOR = -.10


def chain_keys(master_seed, index):
    shelter_evolution.evaluation_keys(master_seed, index)
    return np.asarray(jax.random.split(jax.random.fold_in(
        jax.random.PRNGKey(int(master_seed)), int(index)), 8)).reshape(4, 2, 2)


def _keys(value):
    keys = np.asarray(value)
    if (keys.shape != (4, 2, 2) or keys.dtype != np.uint32
            or len({key.tobytes() for key in keys.reshape(8, 2)}) != 8):
        raise ValueError("each chain requires eight distinct uint32 world and birth keys")
    return keys


def _decoded_birth():
    return dict(track_step_color=np.zeros(0, np.int8), track_step_identified=np.zeros(0, bool),
                track_votes=np.zeros(2, np.int32), track_prediction=np.asarray(-1, np.int32),
                z=np.asarray(0., np.float64), q=np.asarray(.5, np.float32))


def _record_life(flat, keys, rule, frames, lambda_value, c, *, birth=False,
                 reset=False, observer=None):
    if birth:
        observer = belief._birth_observer(flat, keys[1], c)
        decoded = _decoded_birth()
    else:
        observer = social.watch(flat, keys[1], frames, c) if observer is None else observer
        decoded = belief.decode_movie(frames, lambda_value, c)
        if reset:
            decoded["q"] = np.asarray(.5, np.float32)
    life = social.record_pupil(flat, keys[0], keys[1], rule, frames, c,
                               _observer=observer, _record_solo=True,
                               _belief_q=None if birth else jnp.asarray([decoded["q"]]))
    if "solo_signal" not in life:
        raise ValueError("record_pupil must retain actual solo signals before transmission is available")
    grid, position = seed.init_world(keys[0], c)
    life.update(decoded, q_birth=np.asarray(.5, np.float32), readout_enabled=np.asarray(not birth, bool),
                solo_initial_grid=np.asarray(grid, np.uint8),
                solo_initial_position=np.asarray(position, np.int32))
    return life, observer


def life_movie(life, c=CFG):
    """Use only the physical, visible record of this life to reconstruct its film."""
    physical = dict(teacher_initial_grid=life["solo_initial_grid"],
                    teacher_initial_position=life["solo_initial_position"],
                    teacher_grid=life["solo_grid"], teacher_position=life["solo_position"],
                    teacher_signal=life["solo_signal"], teacher_alive_before=life["alive_before"])
    return social.make_movie(physical, c)


def _put(row, prefix, life):
    row.update({prefix + name: value for name, value in life.items()})


def _life(row, prefix):
    return {name[len(prefix):]: value for name, value in row.items() if name.startswith(prefix)}


def evaluate_chain(flat, keys, index, lambda_value, c=CFG):
    social._config(c)
    flat, keys = np.asarray(flat), _keys(keys)
    if (c.T < 6 or flat.shape != (seed.genome_size(c),) or flat.dtype != np.float32
            or not np.isfinite(flat).all() or lambda_value not in belief.LAMBDA_GRID
            or not isinstance(index, Integral) or isinstance(index, (bool, np.bool_)) or index < 0):
        raise ValueError("provide the original float32 genome, frozen lambda and valid chain index")
    rule, swap = int(index) % 2, bool((int(index) // 2) % 2)
    row = dict(keys=keys.copy(), original_genome=flat.copy(), lambda_value=np.asarray(lambda_value, np.float64),
               chain_index=np.asarray(index, np.int32), family_index=np.asarray(index, np.int32),
               rule=np.asarray(rule, np.int32), yoked_swap=np.asarray(swap, bool),
               cohorts=np.asarray(COHORTS, np.int32), steps=np.asarray(c.T, np.int32),
               watch_start=np.asarray(c.T // 2, np.int32), agents=np.asarray(1, np.int32),
               radius=np.asarray(c.R, np.int32), grid_size=np.asarray(c.G, np.int32),
               hidden_units=np.asarray(c.H, np.int32), initial_energy=np.asarray(c.e0, np.float32))
    empty = np.zeros((0, seed.dims(c)[1]), np.float32)
    founder, _ = _record_life(flat, keys[0], rule, empty, lambda_value, c, birth=True)
    _put(row, "founder_", founder)
    founder_movie, row["founder_camera_positions"] = life_movie(founder, c)
    row["founder_movie"] = founder_movie
    previous = founder
    for cohort in range(1, COHORTS + 1):
        movie, camera = life_movie(previous, c)
        row[f"c{cohort}_predecessor_movie"] = movie
        row[f"c{cohort}_camera_positions"] = camera
        row[f"c{cohort}_movie_source_stage"] = np.asarray(cohort - 1, np.int32)
        conditions = social.movie_conditions(movie, swap, c)
        observed_state, predecessor = None, None
        for arm in ARMS:
            frames = founder_movie if arm == "repeated_founder" else conditions["yoked"] if arm == "yoked" else movie
            life, observer = _record_life(flat, keys[cohort], rule, frames, lambda_value, c,
                                          reset=arm == "belief_reset",
                                          observer=observed_state if arm == "belief_reset" else None)
            _put(row, f"c{cohort}_{arm}_", life)
            if arm == "predecessor":
                observed_state, predecessor = observer, life
        previous = predecessor
    validate_chain(row, c)
    return row


def _validate_life(life, c, lambda_value, *, birth=False, reset=False):
    belief._validate_neural(life, "", c, birth=birth)
    grid, position, energy, _ = belief._trajectory(life, "", c)
    signal = cold_donor._array(life, "solo_signal", (c.T, 1))
    if np.any(np.abs(signal) > 1):
        raise ValueError("the public solo signal leaves its physical bounds")
    initial_grid = cold_donor._array(life, "solo_initial_grid", (c.G, c.G), integer=True)
    initial_pos = cold_donor._array(life, "solo_initial_position", (1, 2), integer=True)
    if not np.isin(initial_grid, (0, 1, 2)).all() or np.any(initial_pos < 0) or np.any(initial_pos >= c.G):
        raise ValueError("invalid initial solo layout or body position")
    if float(life["q_birth"]) != .5 or bool(life["readout_enabled"]) != (not birth):
        raise ValueError("every life must begin with neutral belief and its assigned readout")
    expected = _decoded_birth() if birth else belief.decode_movie(life["frames"], lambda_value, c)
    if reset:
        expected["q"] = np.asarray(.5, np.float32)
    for name, value in expected.items():
        if not np.array_equal(life[name], value):
            raise ValueError("task belief differs from this public movie or its assigned reset")
    if birth:
        if any(name.startswith("belief_") for name in life):
            raise ValueError("founder cannot receive an added belief readout")
    else:
        belief._validate_readout(life, "", grid, position, energy, c)


def _same_fields(left, right, names, message):
    if any(not np.array_equal(left[name], right[name]) for name in names):
        raise ValueError(message)


def validate_chain(row, c=CFG):
    social._config(c)
    _keys(row["keys"])
    for name, value in (("cohorts", COHORTS), ("steps", c.T), ("watch_start", c.T // 2),
                        ("agents", 1), ("radius", c.R), ("grid_size", c.G), ("hidden_units", c.H)):
        if cold_donor._scalar(row, name, integer=True) != value:
            raise ValueError("chain configuration differs from the registered protocol")
    index = cold_donor._scalar(row, "chain_index", integer=True)
    if (cold_donor._scalar(row, "family_index", integer=True) != index or int(row["rule"]) != index % 2
            or bool(row["yoked_swap"]) != bool((index // 2) % 2)
            or float(row["initial_energy"]) != np.float32(c.e0)):
        raise ValueError("chain index, rule, yoked assignment or body configuration differs")
    genome = cold_donor._array(row, "original_genome", (seed.genome_size(c),))
    if genome.dtype != np.float32 or float(row["lambda_value"]) not in belief.LAMBDA_GRID:
        raise ValueError("a fixed original float32 genome and frozen lambda are required")
    lambda_value = float(row["lambda_value"])
    founder = _life(row, "founder_")
    _validate_life(founder, c, lambda_value, birth=True)
    founder_movie, camera = life_movie(founder, c)
    if (not np.array_equal(row["founder_movie"], founder_movie)
            or not np.array_equal(row["founder_camera_positions"], camera)):
        raise ValueError("founder movie differs from its own physical solo life")
    previous = founder
    for cohort in range(1, COHORTS + 1):
        movie, camera = life_movie(previous, c)
        if (int(row[f"c{cohort}_movie_source_stage"]) != cohort - 1
                or not np.array_equal(row[f"c{cohort}_predecessor_movie"], movie)
                or not np.array_equal(row[f"c{cohort}_camera_positions"], camera)):
            raise ValueError("the next demonstration must come from the actual preceding solo learner")
        conditions = social.movie_conditions(movie, bool(row["yoked_swap"]), c)
        lives = {arm: _life(row, f"c{cohort}_{arm}_") for arm in ARMS}
        predecessor = lives["predecessor"]
        for arm, life in lives.items():
            _validate_life(life, c, lambda_value, reset=arm == "belief_reset")
            expected = founder_movie if arm == "repeated_founder" else conditions["yoked"] if arm == "yoked" else movie
            if not np.array_equal(life["frames"], expected):
                raise ValueError("cohort frames differ from their assigned source or yoked transform")
            names = ["solo_initial_grid", "solo_initial_position", "birth_sha256"]
            names += ["watch_initial_" + name + "_sha256" for name in ("W", "x", "Wp")]
            _same_fields(life, predecessor, names, "cohort branches must share their own fresh world and newborn")
        reset = lives["belief_reset"]
        names = ["frames", "watch_motor", "watch_denergy", "weight_change_count"]
        names += [stage + "_" + name + "_sha256" for stage in ("watch_initial", "watch_final", "solo_initial")
                  for name in ("W", "x", "Wp")]
        _same_fields(reset, predecessor, names, "belief reset must preserve exactly the predecessor's acquired neural state")
        if cohort == 1:
            # The extra movie provenance field belongs to the chain, not this life.
            names = [name for name in predecessor if name != "movie"]
            if set(names) != set(lives["repeated_founder"]):
                raise ValueError("cohort one predecessor and repeated founder evidence differs")
            _same_fields(predecessor, lives["repeated_founder"], names,
                         "cohort one predecessor and repeated founder must be bit-identical")
        previous = predecessor


def _cfg(row):
    return CFG._replace(T=int(row["steps"]), G=int(row["grid_size"]), R=int(row["radius"]),
                        H=int(row["hidden_units"]), e0=float(row["initial_energy"]))


def summary_input(row):
    """Validate one raw chain, then retain only scalar outcomes and provenance keys."""
    validate_chain(row, _cfg(row))
    result = {name: np.asarray(row[name]).copy() for name in (
        "chain_index", "family_index", "rule", "yoked_swap", "keys", "lambda_value", "steps")}
    result["genome_sha256"] = social._hash(row["original_genome"])
    for prefix in ["founder_"] + [f"c{cohort}_{arm}_" for cohort in range(1, COHORTS + 1) for arm in ARMS]:
        for name in ("first_food", "first_poison", "no_meal", "first_meal_step", "fitness", "weight_change_count", "q"):
            result[prefix + name] = np.asarray(row[prefix + name]).copy()
        result[prefix + "survived"] = np.asarray(row[prefix + "alive_after"][-1, 0], bool)
        result[prefix + "track_votes"] = np.asarray(row[prefix + "track_votes"]).copy()
    return result


def _interval(values):
    stats = cold_donor._mean_se(values)
    mean, se = stats["mean"], stats["se"]
    return {**stats, "lower_2se": mean - 2 * se if se is not None else None,
            "upper_2se": mean + 2 * se if se is not None else None}


def _contrast(first, survival):
    first, survival = _interval(first), _interval(survival)
    defined = first["se"] is not None and survival["se"] is not None
    efficacy = first["lower_2se"] > 0 if defined else None
    safety = survival["lower_2se"] > SURVIVAL_FLOOR if defined else None
    return dict(first_food_difference=first, survival_difference=survival,
                gate=dict(first_food_passed=efficacy, survival_passed=safety,
                          passed=efficacy and safety if defined else None))


def _vectors(rows, prefix):
    fields = ("first_food", "first_poison", "no_meal", "survived", "first_meal_step",
              "fitness", "weight_change_count", "q")
    values = {name: np.asarray([cold_donor._array(row, prefix + name, ()) for row in rows], float) for name in fields}
    for name in ("first_food", "first_poison", "no_meal", "survived"):
        if not np.isin(values[name], (0, 1)).all():
            raise ValueError("summary outcomes must be binary")
    steps = int(rows[0]["steps"])
    if (not np.all(values["first_food"] + values["first_poison"] + values["no_meal"] == 1)
            or np.any(values["first_meal_step"] < 1) or np.any(values["first_meal_step"] > steps + 1)
            or not np.array_equal(values["first_meal_step"] == steps + 1, values["no_meal"].astype(bool))
            or np.any(values["first_meal_step"] != np.floor(values["first_meal_step"]))
            or np.any(np.abs(values["fitness"]) > steps) or np.any(values["weight_change_count"] < 0)
            or np.any(values["weight_change_count"] != np.floor(values["weight_change_count"]))
            or np.any(values["q"] < 0) or np.any(values["q"] > 1)):
        raise ValueError("summary first contact, fitness, watch count or belief leaves valid bounds")
    votes = np.stack([cold_donor._array(row, prefix + "track_votes", (2,), integer=True) for row in rows])
    if np.any(votes < 0) or np.any(votes.sum(axis=1) > steps // 2):
        raise ValueError("summary decoder votes leave the observation budget")
    return values, votes


def _arm_stats(values, votes):
    names = dict(first_food="first_food", first_poison="first_poison", no_meal="no_meal", survival="survived",
                 fitness="fitness", first_meal_step="first_meal_step", changed_weights="weight_change_count",
                 belief_probability="q")
    result = {target: _interval(values[source]) for target, source in names.items()}
    result.update(decoder_coverage=_interval(votes.sum(axis=1) > 0),
                  decoder_strict_coverage=_interval(votes[:, 0] != votes[:, 1]),
                  decoder_votes=_interval(votes.sum(axis=1)),
                  first_food_count=int(values["first_food"].sum()),
                  first_poison_count=int(values["first_poison"].sum()),
                  no_meal_count=int(values["no_meal"].sum()), survivors=int(values["survived"].sum()))
    return result


def summarize(rows):
    """Compute paired chain SE; neither frames nor cohorts are independent samples."""
    rows = [summary_input(row) if "founder_food" in row else row for row in rows]
    if not rows:
        raise ValueError("summarize requires at least one complete assigned chain")
    seen_indices, seen_keys = set(), set()
    lambda_value = float(rows[0]["lambda_value"])
    for row in rows:
        index = cold_donor._scalar(row, "chain_index", integer=True)
        if (index in seen_indices or int(row["family_index"]) != index or int(row["rule"]) != index % 2
                or bool(row["yoked_swap"]) != bool((index // 2) % 2)):
            raise ValueError("summary chain index or assigned condition is duplicated or inconsistent")
        seen_indices.add(index)
        keys = {key.tobytes() for key in _keys(row["keys"]).reshape(8, 2)}
        if keys & seen_keys:
            raise ValueError("independent chains cannot reuse world or birth keys")
        seen_keys.update(keys)
        digest = cold_donor._array(row, "genome_sha256", (32,), integer=True)
        if (digest.dtype != np.uint8 or float(row["lambda_value"]) != lambda_value or lambda_value not in belief.LAMBDA_GRID
                or int(row["steps"]) != int(rows[0]["steps"])
                or not np.array_equal(row["genome_sha256"], rows[0]["genome_sha256"])):
            raise ValueError("all chains must use the same frozen controller and observation budget")
    founder_values, founder_votes = _vectors(rows, "founder_")
    cohorts, vectors = {}, {}
    for cohort in range(1, COHORTS + 1):
        arms = {}
        for arm in ARMS:
            values, votes = _vectors(rows, f"c{cohort}_{arm}_")
            vectors[cohort, arm] = values
            arms[arm] = _arm_stats(values, votes)
        primary = vectors[cohort, "predecessor"]
        contrasts = {arm: _contrast(primary["first_food"] - vectors[cohort, arm]["first_food"],
                                     primary["survived"] - vectors[cohort, arm]["survived"]) for arm in CONTROLS}
        cohorts[str(cohort)] = dict(arms=arms, contrasts=contrasts)
    latest, earliest = vectors[3, "predecessor"], vectors[1, "predecessor"]
    cross = _contrast(latest["first_food"] - earliest["first_food"], latest["survived"] - earliest["survived"])
    cumulative = dict(c3_vs_c1=cross, c3_vs_repeated_founder=cohorts["3"]["contrasts"]["repeated_founder"])
    defined = len(rows) > 1
    efficacy = all(cohorts[str(cohort)]["contrasts"][arm]["gate"]["first_food_passed"]
                   for cohort in range(1, COHORTS + 1) for arm in ("yoked", "belief_reset")) if defined else None
    safety = (all(cohorts[str(cohort)]["contrasts"][arm]["gate"]["survival_passed"]
                  for cohort in range(1, COHORTS + 1) for arm in CONTROLS)
              and cross["gate"]["survival_passed"]) if defined else None
    retained = efficacy and safety if defined else None
    improved = (efficacy and all(value["gate"]["first_food_passed"] for value in cumulative.values())) if defined else None
    cumulative_passed = retained and improved if defined else None
    return dict(protocol=PROTOCOL, chains=len(rows), cohorts=cohorts, selected_lambda=lambda_value,
                agents_per_chain=1 + COHORTS * len(ARMS), assigned_agents=len(rows) * (1 + COHORTS * len(ARMS)),
                rule_counts={str(rule): sum(int(row["rule"]) == rule for row in rows) for rule in (0, 1)},
                founder=_arm_stats(founder_values, founder_votes), cross_cohort=dict(c3_vs_c1=cross),
                cumulative_contrasts=cumulative, survival_floor=SURVIVAL_FLOOR,
                retained_efficacy=dict(passed=retained, first_food_passed=efficacy, survival_passed=safety),
                cumulative_improvement=dict(passed=cumulative_passed, first_food_passed=improved, survival_passed=safety),
                gate=dict(passed=retained, retained_efficacy_passed=retained,
                          cumulative_improvement_passed=cumulative_passed))
