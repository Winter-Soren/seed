"""Post-audit E130 architecture support; descriptive events, never a study gate.

Read only the three completed audited panels. Each independent family retains
both initial rules and all four original animals. No simulator or JAX imports.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
import numpy as np

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
PROTOCOL = "founder-reversal-v1"
PREFIX = "gen200_reversal_"
FIELDS = ("food", "poison", "alive_before", "alive_after")
METRICS = ("quiet_25", "both_food_and_poison_outcomes", "death_before_first_contact")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def family_counts(events):
    """Four predeclared observables in [150,200), without visibility inference."""
    arrays = [np.asarray(events[name]) for name in FIELDS]
    require(all(a.shape == (2, 300, 4) and a.dtype == np.bool_ for a in arrays), "expected complete boolean original-agent events")
    food, poison, before, after = arrays
    require(before[:, 0].all() and np.array_equal(before[:, 1:], after[:, :-1]) and not (after & ~before).any(),
            "birth/death continuity failed")
    require(not (food & poison).any() and not ((food | poison) & ~before).any(), "invalid eating events")
    result = dict(assigned=8, switch_alive=int(before[:, 150].sum()), quiet_at_start=0,
        quiet_after_first_poison=0, quiet_25=0, both_food_and_poison_outcomes=0,
        death_before_first_contact=0, no_contact_while_switch_alive=0, died_in_window=0)
    result["dead_at_switch"] = 8 - result["switch_alive"]
    result["food_events"] = int(food[:, 150:200].sum())
    result["poison_events"] = int(poison[:, 150:200].sum())
    for rule in (0, 1):
        for agent in range(4):
            if not before[rule, 150, agent]:
                continue
            f, p, b, a = (array[rule, 150:200, agent] for array in arrays)
            contact = f | p
            first_contact = int(np.flatnonzero(contact)[0]) if contact.any() else 50
            poison_times = np.flatnonzero(p)
            post_poison = int(poison_times[0]) + 1 if len(poison_times) else 50

            def quiet(start):
                return start + 25 <= 50 and bool((b[start:start + 25] & a[start:start + 25]).all()) and not contact[start:start + 25].any()

            at_start, after_poison = quiet(0), quiet(post_poison)
            deaths = np.flatnonzero(b & ~a)
            result["quiet_at_start"] += int(at_start)
            result["quiet_after_first_poison"] += int(after_poison)
            result["quiet_25"] += int(at_start or after_poison)
            result["both_food_and_poison_outcomes"] += int(f.any() and p.any())
            result["death_before_first_contact"] += int(bool(len(deaths)) and int(deaths[0]) < first_contact)
            result["no_contact_while_switch_alive"] += int(first_contact == 50)
            result["died_in_window"] += int(bool(len(deaths)))
    return result


def ratio(numerator, denominator):
    """Pooled ratio; family-cluster delta SE, including zero-risk families.

    R = mean(A)/mean(B); influence_i = (A_i-R*B_i)/mean(B).
    SE = sample_sd(influence)/sqrt(number of independent families).
    This is not the mean of individual family conditional proportions.
    """
    a, b = np.asarray(numerator, np.float64), np.asarray(denominator, np.float64)
    require(a.ndim == 1 and a.shape == b.shape and len(a) and np.isfinite(a).all() and np.isfinite(b).all()
            and np.all((0 <= a) & (a <= b)), "invalid ratio counts")
    result = dict(numerator=int(a.sum()), denominator=int(b.sum()), families=len(a), mean=None, se=None)
    if b.sum():
        result["mean"] = float(a.sum() / b.sum())
        influence = (a - result["mean"] * b) / b.mean()
        result["se"] = float(influence.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None
    return result


def summarize(families):
    assigned = [row["assigned"] for row in families]
    alive = [row["switch_alive"] for row in families]
    estimates = dict(alive_at_switch=ratio(alive, assigned))
    for metric in METRICS:
        numerator = [row[metric] for row in families]
        estimates[metric] = dict(conditional_on_switch_alive=ratio(numerator, alive), all_assigned=ratio(numerator, assigned))
    return dict(counts={name: sum(row[name] for row in families) for name in families[0] if name != "family"},
                estimates=estimates, per_family=families)


def audit_header(audit):
    require(audit.get("protocol") == PROTOCOL and audit.get("mode") == "final" and audit.get("status") == "passed",
            "requires a complete passed final E130 joint audit")
    panels = audit.get("panels", [])
    require(len(panels) == 3 and sorted(panel.get("index", -1) for panel in panels) == [0, 1, 2], "all three audited lineages required")
    require(isinstance(audit.get("checked_file_sha256"), dict), "joint audit has no file hash manifest")
    for panel in panels:
        measured = panel.get("measurements", {})
        require(measured.get("families") == 64 and measured.get("initial_rule_replays") == 128
                and measured.get("agents_per_rule") == 4, "incomplete original-family audit")
    return sorted(panels, key=lambda panel: panel["index"])


def digest_check(content, expected):
    require(isinstance(expected, str) and hashlib.sha256(content).hexdigest() == expected, "audited file hash mismatch")


def analyze(audit_path):
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    panels = audit_header(audit)
    checked = {}

    def verified(relative):
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), "audited path escapes repository")
        require(relative in audit["checked_file_sha256"], "file absent from passed joint audit: " + relative)
        content = path.read_bytes()
        digest_check(content, audit["checked_file_sha256"][relative])
        checked[relative] = audit["checked_file_sha256"][relative]
        return content

    # Verify completeness and all 192 archive digests before calculating any
    # lineage outcomes. Reading archive bytes for hashing does not select cases.
    for panel in panels:
        index, folder = panel["index"], panel["run"]
        require(folder.startswith("runs/") and folder.endswith("_E130-A" + str(index)), "unexpected final panel")
        state, config = (json.loads(verified(folder + "/" + name)) for name in ("status.json", "config.json"))
        require(state.get("status") == "finished" and state.get("completed_families") == 64
                and state.get("expected_families") == 64, "panel no longer complete")
        require(config.get("protocol") == PROTOCOL and config.get("condition_split") == "final"
                and config.get("seed_index") == index and config.get("expected_families") == 64
                and config.get("eval_seed") == 2026120901 + index, "wrong final panel metadata")
        expected = {f"family_{family:04d}.npz" for family in range(64)}
        require({path.name for path in (ROOT / folder / "families").glob("*.npz")} == expected, "missing or extra family archives")
        for name in sorted(expected):
            verified(folder + "/families/" + name)
    lineages = []
    for panel in panels:
        families = []
        for family in range(64):
            content = verified(panel["run"] + f"/families/family_{family:04d}.npz")
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                require(sum(item.file_size for item in archive.infolist()) <= 1_000_000, "unexpectedly large event archive")
            with np.load(io.BytesIO(content), allow_pickle=False) as raw:
                require(int(raw["base_index"]) == family and np.array_equal(raw["initial_rules"], [0, 1])
                        and int(raw["flip_step"]) == 150, "changed family or window identity")
                counts = family_counts({name: raw[PREFIX + name] for name in FIELDS})
            families.append(dict(family=family, **counts))
        lineages.append(dict(index=panel["index"], run=panel["run"], **summarize(families)))
    return dict(analysis="E130-architecture-feedback-v1", purpose="descriptive architecture decision support; not an E130 gate",
        audit=dict(path=str(audit_path), sha256=hashlib.sha256(audit_bytes).hexdigest(), registered_source=audit.get("registered_source")),
        window=dict(start=150, stop_exclusive=200, agents_per_family=8, initial_rules=2, original_animals_per_rule=4),
        definitions=dict(alive_at_switch="alive_before at turn150",
            quiet_25="switch-alive; all 25 before/after states alive and no contact in [150,175), or immediately after first poison in [150,200) if 25 turns fit",
            both_food_and_poison_outcomes="switch-alive; at least one food and one poison event in [150,200)",
            death_before_first_contact="switch-alive; death transition strictly before first contact in [150,200), or death with no contact in that window"),
        uncertainty="One SE: family-cluster delta method for ratios; both rules and four animals stay in the same independent family. Zero-risk families remain included.",
        lineages=lineages, checked_file_sha256=checked,
        limitations=["Events do not establish item visibility, encounter opportunity, deliberate avoidance, or why contact did not occur.",
            "Food/poison are analysis labels, not exact realized energy-feedback signs; clipping and one-tick feedback timing are not recorded here. No agent uses these labels.",
            "Contact-free does not mean cue-free: unconsumed items and other animals may still be visible.",
            "Conditional estimates describe switch survivors; neither these associations nor this analysis establish a causal architectural advantage."])


def self_check():
    events = dict(food=np.zeros((2, 300, 4), bool), poison=np.zeros((2, 300, 4), bool),
                  alive_before=np.ones((2, 300, 4), bool), alive_after=np.ones((2, 300, 4), bool))
    for rule, agent, turn in ((0, 1, 150), (0, 2, 174), (0, 3, 175), (1, 0, 151), (1, 3, 150)):
        events["poison"][rule, turn, agent] = True
    events["food"][0, 160, 3] = events["food"][1, 150, 0] = True
    for rule, agent, turn in ((1, 1, 149), (1, 2, 150), (1, 3, 150)):
        events["alive_after"][rule, turn:, agent] = False
        events["alive_before"][rule, turn + 1:, agent] = False
    result = family_counts(events)
    require(all(result[name] == value for name, value in dict(assigned=8, switch_alive=7, dead_at_switch=1,
        quiet_at_start=1, quiet_after_first_poison=3, quiet_25=4, both_food_and_poison_outcomes=2,
        death_before_first_contact=1, no_contact_while_switch_alive=2, died_in_window=2, food_events=2, poison_events=5).items()),
        "independent hand-counted contact/death/boundary fixture")
    conditional = ratio([1, 1], [1, 3])
    require(conditional["mean"] == .5 and np.isclose(conditional["se"], .25), "conditional ratio must preserve cluster covariance")
    assigned = ratio([1, 3], [8, 8])
    require(assigned["mean"] == .25 and np.isclose(assigned["se"], .125), "all-assigned family SE")
    require(ratio([0, 0], [0, 0])["mean"] is None, "undefined conditional denominator must not become zero")
    for action in (lambda: audit_header({"protocol": PROTOCOL, "mode": "development", "status": "passed"}),
                   lambda: digest_check(b"changed", hashlib.sha256(b"original").hexdigest())):
        try:
            action()
        except ValueError:
            continue
        raise AssertionError("incomplete audit or changed bytes accepted")
    return dict(status="passed", checks=6, scientific_inputs_opened=False, simulator_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        require(args.audit is None and args.out is None, "synthetic self-check cannot open scientific inputs or write a report")
        print(json.dumps(self_check(), indent=2, allow_nan=False))
        return
    if args.audit is None or args.out is None:
        parser.error("--audit and --out are required after the final joint audit passes")
    audit_path, output = ((ROOT / path).resolve() for path in (args.audit, args.out))
    require(output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "runs")
            and not output.is_relative_to(ROOT / ".git") and not output.exists(), "report must be a new non-run repository file")
    report = analyze(audit_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(output))


if __name__ == "__main__":
    main()
