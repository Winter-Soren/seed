"""E138 matched native selection on a focal choice or free first-item foraging."""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from . import encounter_diagnostic as encounter
from . import replay_selection as replay
from . import scene_transfer as scene
from . import seed


PROTOCOL = "solo-selection-v1"
CFG = encounter.CFG
ASSIGNMENT_TAG = 0xE138
GENOTYPES = ("parent", "background", "solo")
HISTORIES = ("stationary", "reversal")
VIEWS = ("background", "solo")
KEY_ROLES = ("script", "birth", "chamber", "background", "probe", "solo_layout", "solo_action")
TRAIN_KEY_ROLES = ("script", "chamber", "background", "probe", "solo_layout", "solo_action")
TRAIN_CELLS = replay.TRAIN_CELLS.copy()
GRID_STEPS = replay.GRID_STEPS
SOLO_AGENT_FIELDS = ("action", "sampled_action", "food", "poison", "alive_after", "item", "energy",
                     "incoming_denergy")
SOLO_FLAGS = ("forced_action", "hebbian_enabled", "predictor_enabled")


def training_batch(world_keys):
    """Materialize rule-blind apparatus and ordinary grids for four families."""
    world_keys = replay._keys(world_keys, 4)
    role_keys = np.stack([np.asarray(jax.random.split(key, len(TRAIN_KEY_ROLES))) for key in world_keys])
    schedule_key = np.asarray(jax.random.fold_in(world_keys[0], ASSIGNMENT_TAG))
    cells = TRAIN_CELLS[np.asarray(jax.random.permutation(schedule_key, len(TRAIN_CELLS)))]
    scripts = np.zeros((4, 32, 5), np.int32)
    backgrounds, grids, positions = [], [], []
    for index, (pre, _) in enumerate(cells):
        script = encounter.contact_script(role_keys[index, 0], int(pre))
        scripts[index, :len(script)] = script
        backgrounds.append(np.asarray(scene._background()(jnp.asarray(role_keys[index, 2]))))
        grid, position = seed.init_world(jnp.asarray(role_keys[index, 4]), CFG)
        grids.append(np.asarray(grid))
        positions.append(np.asarray(position))
    return dict(world_keys=world_keys.copy(), role_keys=role_keys, schedule_key=schedule_key,
                scripts=scripts, backgrounds=np.stack(backgrounds), solo_grids=np.stack(grids),
                solo_positions=np.stack(positions), pre_contacts=cells[:, 0], gap=cells[:, 1])


def first_item_score(food, poison):
    """Score the first actual contact along the final axis, including lethal contacts."""
    contact = food | poison
    first = jnp.argmax(contact, axis=-1)[..., None]
    signed = food.astype(jnp.float32) - poison.astype(jnp.float32)
    return jnp.take_along_axis(signed, first, axis=-1)[..., 0]


def _score_family(flat, script, birth_key, chamber_key, background, probe_key,
                  solo_grid, solo_position, solo_key, pre_contacts, gap):
    probe_keys = jax.random.split(probe_key, len(encounter.PLACEMENTS))

    def initial_rule(rule):
        birth = encounter._birth(flat, "plastic", birth_key, chamber_key, CFG)
        boundary, prior = encounter._chamber("plastic", CFG)(flat, birth, script[:pre_contacts], rule)
        pre_counts = jnp.stack((prior["food"].sum(), prior["poison"].sum())).astype(jnp.float32)
        pre_bounds = jnp.stack((prior["unclipped_energy"].min(), prior["unclipped_energy"].max()))

        def history_score(history):
            target = jnp.bitwise_xor(rule, history)
            delayed, post_counts, bounds = replay._training_history(
                flat, boundary, script[pre_contacts:pre_contacts + 16], target, gap)
            _, first = scene._first_decisions()(flat, delayed, background, probe_keys, target)
            # Both probes inherit the unprobed brain. The first-choice action
            # never alters the fresh ordinary-grid trajectory.
            fresh = encounter._fresh_body(delayed, solo_grid, solo_position, solo_key, CFG)
            _, solo = encounter._free_grid("plastic", CFG, steps=GRID_STEPS)(flat, fresh, target)
            compact = {"solo_" + name: solo[name][:, 0] for name in SOLO_AGENT_FIELDS}
            compact.update({"solo_" + name: solo[name] for name in SOLO_FLAGS})
            compact["solo_logits"] = solo["logits"][:, 0, :]
            return dict(probabilities=first["probabilities"][1, :, 0, :],
                        logits=first["logits"][1, :, 0, :],
                        sampled_action=first["sampled_action"][1, :, 0],
                        probe_action_keys=first["action_key"][1],
                        counts=jnp.stack((pre_counts, post_counts)), survived=delayed["alive"][0],
                        exposure_energy_bounds=jnp.stack((jnp.minimum(pre_bounds[0], bounds[0]),
                                                         jnp.maximum(pre_bounds[1], bounds[1]))),
                        **compact)
        return jax.lax.map(history_score, jnp.arange(2, dtype=jnp.int32))

    result = jax.lax.map(initial_rule, jnp.arange(2, dtype=jnp.int32))
    probabilities = result["probabilities"]
    placements = jnp.asarray(encounter.PLACEMENTS)
    targets = jnp.bitwise_xor(jnp.arange(2)[:, None], jnp.arange(2)[None, :])
    good = placements[:, targets].transpose(1, 2, 0)[..., None]
    bad = placements[:, 1 - targets].transpose(1, 2, 0)[..., None]
    food = jnp.take_along_axis(probabilities, good, axis=-1)[..., 0]
    poison = jnp.take_along_axis(probabilities, bad, axis=-1)[..., 0]
    local_score = (food - poison).mean()
    solo_score = first_item_score(result["solo_food"], result["solo_poison"]).mean()
    compact = {name: value for name, value in result.items() if name.startswith("solo_")}
    return dict(view_fitness=jnp.stack((local_score, solo_score)), probabilities=probabilities,
                logits=result["logits"], sampled_action=result["sampled_action"],
                probe_action_keys=result["probe_action_keys"], counts=result["counts"].mean((0, 1)),
                survived=result["survived"].min(),
                exposure_energy_bounds=jnp.stack((result["exposure_energy_bounds"][..., 0].min(),
                                                 result["exposure_energy_bounds"][..., 1].max())),
                **compact)


@lru_cache(None)
def _population_cell(pre_contacts, gap):
    # Initialize cached factories on the host so nested scans cannot retain tracers.
    encounter._unravel(CFG)
    encounter._chamber("plastic", CFG)
    replay._empty_gap(gap)
    scene._first_decisions()
    encounter._free_grid("plastic", CFG, steps=GRID_STEPS)

    @jax.jit
    def run(candidates, script, birth_keys, chamber_key, background, probe_key,
            solo_grid, solo_position, solo_key):
        return jax.lax.map(lambda pair: _score_family(
            pair[0], script, pair[1], chamber_key, background, probe_key,
            solo_grid, solo_position, solo_key, pre_contacts, gap), (candidates, birth_keys))
    return run


def make_evaluator(view, c=CFG):
    """Return the same compiled evidence for both objectives, then route one scalar."""
    encounter._config(c)
    if view not in VIEWS:
        raise ValueError("training objective must be background or solo")
    view_index = VIEWS.index(view)

    def evaluate(candidates, batch, birth_keys):
        candidates, birth_keys = np.asarray(candidates), np.asarray(birth_keys)
        if (candidates.ndim != 2 or not len(candidates) or candidates.shape[1:] != (155,)
                or candidates.dtype != np.float32 or not np.isfinite(candidates).all()
                or birth_keys.shape != (len(candidates), 4, 2) or birth_keys.dtype != np.uint32):
            raise ValueError("candidate and antithetic birth arrays have invalid dimensions")
        if sorted(zip(np.asarray(batch["pre_contacts"]).tolist(), np.asarray(batch["gap"]).tolist())) != list(map(tuple, TRAIN_CELLS)):
            raise ValueError("training must contain each registered schedule cell once")
        expected = {"role_keys": ((4, 6, 2), np.uint32), "scripts": ((4, 32, 5), np.int32),
                    "backgrounds": ((4, c.G, c.G), np.int32),
                    "solo_grids": ((4, c.G, c.G), np.int32), "solo_positions": ((4, 1, 2), np.int32)}
        if any(np.shape(batch[name]) != shape or np.asarray(batch[name]).dtype != dtype
               for name, (shape, dtype) in expected.items()):
            raise ValueError("training materialization differs from the four-family contract")
        records = []
        for index in range(4):
            pre, gap = int(batch["pre_contacts"][index]), int(batch["gap"][index])
            records.append(_population_cell(pre, gap)(
                jnp.asarray(candidates), jnp.asarray(batch["scripts"][index]), jnp.asarray(birth_keys[:, index]),
                jnp.asarray(batch["role_keys"][index, 1]), jnp.asarray(batch["backgrounds"][index]),
                jnp.asarray(batch["role_keys"][index, 3]), jnp.asarray(batch["solo_grids"][index]),
                jnp.asarray(batch["solo_positions"][index]), jnp.asarray(batch["role_keys"][index, 5])))
        result = jax.tree.map(lambda *values: jnp.stack(values, axis=1), *records)
        bounds = np.asarray(result["exposure_energy_bounds"])
        if (not np.isfinite(np.asarray(result["view_fitness"])).all() or not np.isfinite(bounds).all()
                or bounds[..., 0].min() <= 0 or bounds[..., 1].max() >= c.e_max
                or not np.all(np.asarray(result["survived"]) == 1)):
            raise ValueError("candidate replay violated finite scores or guaranteed survival")
        if (not np.all(np.asarray(result["solo_forced_action"]) == -1)
                or not np.array_equal(result["solo_action"], result["solo_sampled_action"])
                or not np.all(np.asarray(result["solo_hebbian_enabled"]))
                or not np.all(np.asarray(result["solo_predictor_enabled"]))):
            raise ValueError("ordinary-grid scoring requires every native voluntary action and update")
        return dict(result, fitness=result["view_fitness"][..., view_index])
    return evaluate
