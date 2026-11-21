"""E117: paired newborns after one cold donor life, with no evolution.

One row is one independent world family. The public lineage implementation owns
development, plasticity and archive transfer; this module only fixes the pairing
and measures its output. No state is carried between rows. NPZ rows contain plain
numeric arrays and booleans, so reading evidence never requires pickle.
"""
import hashlib
from numbers import Integral

import jax
import numpy as np

from seed import lineage, seed
from seed.ecology import EcologyV1


PROTOCOL = "cold-donor-v1"
FINAL_SEEDS = (2026112101, 2026112102, 2026112103)
DEVELOPMENT_SEED = 42
FAMILIES = 256
ECOLOGY = EcologyV1(movement_energy_cost=.01, seasonal_energy_cost=.02)
ARMS = ("donor", "inherited", "fresh")
EVENTS = ("food", "poison", "alive_before", "alive_after", "sheltered")


def family_keys(master_seed, index):
    """Return donor world/birth, then target world/birth threefry keys.

    Folding the family index before splitting creates an independent lineage;
    the target arms share both target keys, including their action RNG stream.
    """
    if (not isinstance(master_seed, Integral) or isinstance(master_seed, (bool, np.bool_))
            or not 0 <= master_seed < 2 ** 32 or not isinstance(index, Integral)
            or isinstance(index, (bool, np.bool_)) or not 0 <= index < 2 ** 32):
        raise ValueError("seed and family index must be uint32 integers")
    return np.asarray(jax.random.split(jax.random.fold_in(
        jax.random.PRNGKey(int(master_seed)), int(index)), 4))


def first_entry(initial_sheltered, sheltered, alive_after):
    """First living arrival: birth=0, turn=1..T, nonarrival (including death)=T+1."""
    initially = np.asarray(initial_sheltered, bool)
    arriving = np.asarray(sheltered, bool) & np.asarray(alive_after, bool)
    turns = np.arange(1, arriving.shape[0] + 1, dtype=np.int32)[:, None]
    return np.where(initially, 0, np.where(arriving, turns, arriving.shape[0] + 1).min(0)).astype(np.int32)


def _fingerprints(weights):
    return np.stack([np.frombuffer(hashlib.sha256(agent.tobytes()).digest(), np.uint8)
                     for agent in weights])


def _record_arm(raw, prefix, position, result):
    if "persistence" in raw:
        raise ValueError("E117 requires persistence disabled in every life")
    for name in EVENTS:
        source = raw["ecology"] if name == "sheltered" else raw["events"]
        value = np.asarray(source[name])
        if not np.isin(value, (0, 1)).all():
            raise ValueError(f"nonbinary {prefix} {name}")
        result[prefix + "_" + name] = value.astype(bool)
    for name in ("shelter", "initial_sheltered", "energy_lost"):
        result[prefix + "_" + name] = np.asarray(raw["ecology"][name])
    result[prefix + "_initial_position"] = np.asarray(position)
    result[prefix + "_fitness"] = np.asarray(raw["fitness"])
    for name in ("birth", "initial", "terminal"):
        result[prefix + "_" + name + "_sha256"] = np.stack([
            np.frombuffer(bytes.fromhex(digest), np.uint8)
            for digest in raw["weights"][name + "_W_sha256"]])
    result[prefix + "_changed_connections"] = np.asarray(raw["weights"]["changed_connections"])
    result[prefix + "_mean_abs_weight_change"] = np.asarray(raw["weights"]["mean_abs_change"])
    result[prefix + "_first_entry"] = first_entry(
        result[prefix + "_initial_sheltered"], result[prefix + "_sheltered"],
        result[prefix + "_alive_after"])


def evaluate_family(g, keys, rule, c=seed.Cfg(), ecology=ECOLOGY):
    """Run two independent newborn cohorts with actual ``collective_lives``.

    This deliberately retains the public path's paired empty-archive donor arm;
    it runs four rollouts, two of which must be identical. The single donor life
    updates the recipient archive once. Recipient learning is never fed back to
    another family. Defaults are the final protocol; small configurations support
    development-only verification without touching any held-out keys.
    """
    keys = np.asarray(keys)
    if keys.shape != (4, 2) or keys.dtype != np.uint32 or len({key.tobytes() for key in keys}) != 4:
        raise ValueError("provide four distinct uint32[2] keys")
    if not isinstance(ecology, EcologyV1):
        raise ValueError("cold donor measurement requires ecology-v1")
    result = lineage.collective_lives(
        g, keys[[0, 2]], keys[[1, 3]], [rule, rule], c,
        ecologies=(ecology, ecology), plastic=True, persistence=None)
    if result["completed_lives"] != 2 or "persistence" in result:
        raise ValueError("expected exactly two persistence-free cohorts")
    donor, target = result["lives"]
    if np.any(donor["archive_before"]) or not np.array_equal(target["archive_before"], donor["archive_after"]):
        raise ValueError("archive must start empty and receive exactly one donor life")
    for name in ("food", "poison", "alive_before", "alive_after"):
        if not np.array_equal(donor["inherited"]["events"][name], donor["fresh"]["events"][name]):
            raise ValueError("the empty-archive donor arms must be identical")
    for name, value in donor["inherited"]["weights"].items():
        if not np.array_equal(value, donor["fresh"]["weights"][name]):
            raise ValueError("empty-archive donor weights differ")

    _, sensors, motors, neurons, _ = seed.dims(c, ecology=ecology)
    birth, mask = lineage._birth(g, keys[3], c, ecology)
    fresh = np.clip(birth, -c.wmax, c.wmax)
    inherited = fresh.copy()
    recipient = inherited[:, neurons - motors:, :sensors]
    overlay = donor["archive_after"] * mask[:, neurons - motors:, :sensors]
    np.copyto(recipient, np.clip(recipient + overlay, -c.wmax, c.wmax), where=overlay != 0)
    row = dict(keys=keys.copy(), rule=np.asarray(rule, np.int32), steps=np.asarray(c.T, np.int32),
               agents=np.asarray(c.M, np.int32), cold_start=np.asarray(c.T // 2, np.int32),
               weight_bound=np.asarray(c.wmax), archive_before=donor["archive_before"].copy(),
               archive_after=donor["archive_after"].copy(), donor_delta=donor["donor_delta"].copy(),
               donor_mask=donor["donor_mask"].astype(bool), donor_counts=donor["donor_counts"].copy(),
               recipient_mask=mask[:, neurons - motors:, :sensors].astype(bool),
               recipient_birth=fresh[:, neurons - motors:, :sensors].copy(),
               recipient_initial_delta=(inherited - fresh)[:, neurons - motors:, :sensors].copy())
    _, donor_position = seed.init_world(keys[0], c)
    _, target_position = seed.init_world(keys[2], c)
    _record_arm(donor["inherited"], "donor", donor_position, row)
    _record_arm(target["inherited"], "inherited", target_position, row)
    _record_arm(target["fresh"], "fresh", target_position, row)
    for arm, actual in (("inherited", inherited), ("fresh", fresh)):
        if not np.array_equal(row[arm + "_initial_sha256"], _fingerprints(actual)):
            raise ValueError("recipient initial weights disagree with the exact archive overlay")
    # rollout's birth_W is its JIT-developed, pre-override value. The actual
    # initial_W comes from collective_lives' eager development and can differ
    # by rounding. Validate actual initial hashes above and paired birth hashes
    # below, without equating these two arithmetic paths.
    validate_family(row, c)
    return row


def _scalar(row, name, *, integer=False):
    value = np.asarray(row[name])
    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):
        raise ValueError(f"invalid {name} scalar")
    if integer and (value.dtype.kind not in "iu" or value < 0):
        raise ValueError(f"invalid {name} integer")
    return int(value) if integer else float(value)


def _array(row, name, shape, *, binary=False, integer=False):
    value = np.asarray(row[name])
    if value.shape != shape or value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        raise ValueError(f"invalid {name} shape or values")
    if binary and not np.isin(value, (0, 1)).all():
        raise ValueError(f"nonbinary {name}")
    if integer and value.dtype.kind not in "iu":
        raise ValueError(f"noninteger {name}")
    return value.astype(bool) if binary else value


def validate_family(row, c=None):
    """Validate saved event timing, matched starts, donor update and all metrics."""
    steps, agents = (_scalar(row, name, integer=True) for name in ("steps", "agents"))
    cold = _scalar(row, "cold_start", integer=True)
    rule = _scalar(row, "rule", integer=True)
    bound = _scalar(row, "weight_bound")
    if steps < 1 or agents < 1 or cold != steps // 2 or rule not in (0, 1) or bound <= 0:
        raise ValueError("invalid family dimensions, rule or weight bound")
    if c is not None and (steps != c.T or agents != c.M or bound != c.wmax):
        raise ValueError("family differs from its configuration")
    keys = _array(row, "keys", (4, 2), integer=True)
    if keys.dtype != np.uint32 or len({key.tobytes() for key in keys}) != 4:
        raise ValueError("family keys are not distinct threefry keys")
    for arm in ARMS:
        prefix = arm + "_"
        f, p, before, after, sheltered = [_array(row, prefix + name, (steps, agents), binary=True)
                                          for name in EVENTS]
        initially = _array(row, prefix + "initial_sheltered", (agents,), binary=True)
        if (not before[0].all() or np.any(f & p) or np.any((f | p | sheltered) & ~before)
                or np.any(after & ~before) or not np.array_equal(before[1:], after[:-1])):
            raise ValueError(f"inconsistent {arm} events or death timing")
        loss = _array(row, prefix + "energy_lost", (steps, agents))
        if np.any(loss < 0) or np.any(loss[~before] != 0):
            raise ValueError(f"invalid {arm} energy loss")
        fitness = _scalar(row, prefix + "fitness")
        if fitness != float((f.sum(0, dtype=np.int64) - p.sum(0, dtype=np.int64)).mean()):
            raise ValueError(f"{arm} fitness disagrees with original-agent events")
        arrival = _array(row, prefix + "first_entry", (agents,), integer=True)
        if not np.array_equal(arrival, first_entry(initially, sheltered, after)):
            raise ValueError(f"{arm} first entry disagrees with living arrivals")
        shelter = np.asarray(row[prefix + "shelter"])
        if shelter.ndim != 2 or shelter.shape[0] != shelter.shape[1] or shelter.shape[0] < 1:
            raise ValueError("shelter must be a square map")
        shelter = _array(row, prefix + "shelter", shelter.shape, binary=True)
        position = _array(row, prefix + "initial_position", (agents, 2), integer=True)
        if (np.any(position < 0) or np.any(position >= shelter.shape[0])
                or not np.array_equal(initially, shelter[position[:, 0], position[:, 1]])):
            raise ValueError("initial shelter occupancy disagrees with the recorded position")
        if c is not None and shelter.shape != (c.G, c.G):
            raise ValueError("shelter map differs from configuration")
        for name in ("birth", "initial", "terminal"):
            hashes = _array(row, prefix + name + "_sha256", (agents, 32), integer=True)
            if hashes.dtype != np.uint8:
                raise ValueError("fingerprints must be uint8 SHA256 bytes")
        for name in ("changed_connections", "mean_abs_weight_change"):
            value = _array(row, prefix + name, (agents,), integer=name == "changed_connections")
            if np.any(value < 0):
                raise ValueError("weight changes cannot be negative")
    for field in ("shelter", "initial_position", "initial_sheltered", "birth_sha256"):
        if not np.array_equal(row["inherited_" + field], row["fresh_" + field]):
            raise ValueError("recipient arms must have matched worlds, births and outside eligibility")

    archive = np.asarray(row["archive_before"])
    if archive.ndim != 2 or not archive.size:
        raise ValueError("archive must be a nonempty motor by sensor matrix")
    shape = archive.shape
    archive = _array(row, "archive_before", shape)
    final = _array(row, "archive_after", shape)
    delta = _array(row, "donor_delta", (agents,) + shape)
    mask = _array(row, "donor_mask", (agents,) + shape, binary=True)
    counts = _array(row, "donor_counts", shape)
    if np.any(archive) or np.any(delta[~mask] != 0) or not np.array_equal(counts, mask.sum(0)):
        raise ValueError("donor archive must start empty and average all masked original donors")
    expected = np.clip(delta.sum(0) / np.maximum(counts.astype(delta.dtype), 1), -bound, bound)
    if not np.array_equal(final, expected):
        raise ValueError("archive differs from masked donor update")
    recipient_mask = _array(row, "recipient_mask", (agents,) + shape, binary=True)
    birth = _array(row, "recipient_birth", (agents,) + shape)
    inherited_delta = _array(row, "recipient_initial_delta", (agents,) + shape)
    expected = birth.copy()
    overlay = final * recipient_mask
    np.copyto(expected, np.clip(expected + overlay, -bound, bound), where=overlay != 0)
    if (np.any(np.abs(birth) > bound) or np.any(birth[~recipient_mask] != 0)
            or not np.array_equal(inherited_delta, expected - birth)):
        raise ValueError("recipient changes disagree with bounded archive inheritance")
    if c is not None and shape != (seed.dims(c, ecology=ECOLOGY)[2], seed.dims(c, ecology=ECOLOGY)[1]):
        raise ValueError("archive layout differs from configuration")
    return None


def _mean_se(values):
    values = np.asarray(values, np.float64)
    return dict(mean=float(values.mean()), se=None if len(values) < 2 else
                float(values.std(ddof=1) / np.sqrt(len(values))))


def _ratio_se(numerators, denominators):
    """Ratio of pooled agent totals; cluster influence SE over world families."""
    y, z = np.asarray(numerators, np.float64), np.asarray(denominators, np.float64)
    if z.sum() == 0:
        return dict(mean=None, se=None)
    estimate = float(y.sum() / z.sum())
    influence = (y - estimate * z) / z.mean()
    return dict(mean=estimate, se=None if len(y) < 2 else
                float(influence.std(ddof=1) / np.sqrt(len(y))))


def summarize(rows):
    """Reconstruct compact paired statistics from raw independently saved families.

    Error bars always use independent families, never individual agents. Latency
    and arrival fractions pool outside-born agents using a ratio estimator; no
    survivor-only restriction is applied. Gate fields are undecided with fewer
    than two independent families or no outside-born recipients. The runner must
    additionally require all 256 preregistered families before a final decision.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("cannot summarize no completed families")
    for row in rows:
        validate_family(row)
    steps, agents, cold = (int(rows[0][name]) for name in ("steps", "agents", "cold_start"))
    if any(any(int(row[name]) != value for name, value in
               (("steps", steps), ("agents", agents), ("cold_start", cold))) for row in rows):
        raise ValueError("families must share dimensions and cold timing")
    seen = set()
    for row in rows:
        for key in np.asarray(row["keys"]):
            signature = key.tobytes()
            if signature in seen:
                raise ValueError("families must not reuse random keys")
            seen.add(signature)
    outside = np.stack([~np.asarray(row["fresh_initial_sheltered"], bool) for row in rows])
    denominator = outside.sum(1)
    by_arm = {}
    for arm in ARMS:
        eligible = (~np.stack([np.asarray(row["donor_initial_sheltered"], bool) for row in rows])
                    if arm == "donor" else outside)
        arrivals = np.stack([row[arm + "_first_entry"] for row in rows])
        living = np.stack([row[arm + "_alive_after"] for row in rows]).astype(bool)
        before = np.stack([row[arm + "_alive_before"] for row in rows]).astype(bool)
        occupied = np.stack([row[arm + "_sheltered"] for row in rows]).astype(bool)
        cold_occupied = occupied[:, cold:] & living[:, cold:]
        exposed = before[:, cold:] & ~occupied[:, cold:]
        by_arm[arm] = dict(
            latency=_ratio_se((arrivals * eligible).sum(1), eligible.sum(1)),
            shelter_arrival=_ratio_se(((arrivals <= steps) & eligible).sum(1), eligible.sum(1)),
            survival=_mean_se(living[:, -1].mean(1)),
            fitness=_mean_se([float(row[arm + "_fitness"]) for row in rows]),
            cold_sheltered_fraction=_mean_se(cold_occupied.mean((1, 2))),
            outdoor_cold_agent_steps=_mean_se(exposed.sum((1, 2))),
            extra_energy_lost=_mean_se([np.asarray(row[arm + "_energy_lost"]).sum(dtype=np.float64) for row in rows]),
            deaths=int((~living[:, -1]).sum()), cold_exposed_agents=int(exposed.any(1).sum()),
            cold_exposed_agent_steps=int(exposed.sum()),
            living_arrivals=int(((arrivals <= steps) & eligible).sum()),
            outside_born_agents=int(eligible.sum()))
    fresh_latency = np.stack([row["fresh_first_entry"] for row in rows])
    inherited_latency = np.stack([row["inherited_first_entry"] for row in rows])
    latency = _ratio_se(((fresh_latency - inherited_latency) * outside).sum(1), denominator)
    survival = _mean_se([np.asarray(row["inherited_alive_after"])[-1].mean()
                         - np.asarray(row["fresh_alive_after"])[-1].mean() for row in rows])
    decided = latency["se"] is not None and survival["se"] is not None
    latency_passed = bool(latency["mean"] > 2 * latency["se"]) if decided else None
    survival_passed = bool(survival["mean"] >= -2 * survival["se"]) if decided else None
    return dict(families=len(rows), agents_per_family=agents, assigned_agents=len(rows) * agents,
                outside_born_agents=int(outside.sum()),
                rule_counts={str(rule): sum(int(row["rule"]) == rule for row in rows) for rule in (0, 1)},
                **by_arm, latency_improvement=latency, survival_difference=survival,
                archive=dict(changed_entries=_mean_se([np.count_nonzero(row["archive_after"]) for row in rows]),
                             mean_abs_update=_mean_se([np.abs(row["archive_after"]).mean(dtype=np.float64) for row in rows]),
                             recipients_changed=int(sum(np.any(row["recipient_initial_delta"] != 0, axis=(1, 2)).sum() for row in rows))),
                gate=dict(passed=latency_passed and survival_passed if decided else None,
                          latency_passed=latency_passed, survival_passed=survival_passed))
