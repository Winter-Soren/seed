"""Independent E134 choice, physics, optimizer and paired-family reconstruction.

Only JAX random primitives and explicit optimizer arithmetic are replayed. The
auditor never imports Seed's simulator, evaluator or production optimizer.
"""
import argparse
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_encounter_diagnostic as encounter
from audit_social_observer_evolution import average_ranks
from audit_shelter_evolution import close

base, source_bytes = encounter.base, encounter.source_bytes
require, same, subset, typed = base.require, base.same, base.subset, base.typed
exact, fingerprint, prefixed = encounter.exact, encounter.fingerprint, encounter.prefixed
PROTOCOL = "replay-selection-v1"
CFG, MOVES = encounter.CFG, encounter.MOVES
NEURAL = ("W", "x", "Wp", "mask")
PLACEMENTS = np.asarray([(a, b) for a in range(1, 5) for b in range(1, 5) if a != b], np.int32)
SOURCE_REQUIRED = {"seed/seed.py", "seed/encounter_diagnostic.py", "seed/replay_selection.py",
                   "scripts/social_observer_evolution.py", "scripts/run_replay_selection.py"}
GENOTYPES = ("parent", "descendant")
HISTORIES = ("stationary", "reversal")
FEEDBACK = ("informative", "yoked")


def physical_close(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
        and np.all(np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                   <= 8 * np.finfo(np.float32).eps * np.maximum(1., np.abs(expected))), label)


def signed_choice(probabilities, target):
    probabilities = np.asarray(probabilities)
    require(probabilities.shape[-2:] == (12, 5), "all twelve placements and five actions")
    shape = (1,)*(probabilities.ndim-2)+(12,1)
    good = np.take_along_axis(probabilities, PLACEMENTS[:, target, None].reshape(shape), axis=-1)[..., 0]
    bad = np.take_along_axis(probabilities, PLACEMENTS[:, 1-target, None].reshape(shape), axis=-1)[..., 0]
    return (good.astype(np.float64) - bad.astype(np.float64)).mean(-1)


def statistics(values):
    values = np.asarray(values, np.float64)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), "independent finite family contrasts")
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return dict(mean=float(values.mean()), se=se, n=len(values), lower_2se=float(values.mean()-2*se))


def passes(value, floor=0.):
    return bool(value["mean"] >= floor and value["mean"] > 2*value["se"])


def summary(rows):
    """One observation per family after its paired rules and clones are averaged."""
    old = np.stack([row["old"] for row in rows]).mean(2)
    local = np.stack([row["local"] for row in rows]).mean(2)
    solo = {name: np.stack([row[name] for row in rows]).mean(2)
            for name in ("Y", "first_food", "first_poison", "no_contact", "survival", "latency")}
    require(old.shape == (len(rows), 2) and local.shape == (len(rows), 2, 2, 2), "paired family/local dimensions")
    local_gain = local[:, 1, 1, 0]-local[:, 0, 1, 0]
    yoked_gain = local[:, 1, 1, 1]-local[:, 0, 1, 1]
    g_r = solo["Y"][:, 1, 1]-solo["Y"][:, 0, 1]
    g_t = solo["Y"][:, 1, 0]-solo["Y"][:, 0, 0]
    vectors = dict(old_descendant=old[:, 1], reversal_local_descendant=local[:, 1, 1, 0],
        informative_history_effect=local[:, 1, 1, 0]+local[:, 1, 0, 0],
        local_gain=local_gain, feedback_gain=local_gain-yoked_gain,
        reversal_solo_descendant=solo["Y"][:, 1, 1], solo_gain=g_r, adjusted_solo_gain=g_r-g_t,
        old_safeguard=old[:, 1]-old[:, 0], stationary_local_safeguard=local[:, 1, 0, 0]-local[:, 0, 0, 0],
        stationary_solo_safeguard=g_t,
        stationary_survival_safeguard=solo["survival"][:, 1, 0]-solo["survival"][:, 0, 0],
        reversal_survival_safeguard=solo["survival"][:, 1, 1]-solo["survival"][:, 0, 1])
    measures = {name: statistics(value) for name, value in vectors.items()}
    floors = dict(old_descendant=0., reversal_local_descendant=.10, informative_history_effect=.10,
        local_gain=.05, feedback_gain=.025, reversal_solo_descendant=0., solo_gain=.05, adjusted_solo_gain=.05)
    criteria = {name: passes(measures[name], floor) for name, floor in floors.items()}
    criteria.update({name: bool(value["lower_2se"] > -.05) for name, value in measures.items() if name.endswith("safeguard")})
    arms = {}
    for genotype, label in enumerate(GENOTYPES):
        arms[label] = dict(old_preference=statistics(old[:, genotype]), histories={})
        for history, hlabel in enumerate(HISTORIES):
            arms[label]["histories"][hlabel] = dict(
                local={flabel: statistics(local[:, genotype, history, feedback]) for feedback, flabel in enumerate(FEEDBACK)},
                solo={name: statistics(value[:, genotype, history]) for name, value in solo.items()})
            good = sum(float(row["first_food"][genotype,:,history].sum()) for row in rows)
            bad = sum(float(row["first_poison"][genotype,:,history].sum()) for row in rows)
            arms[label]["histories"][hlabel]["conditional_contact_accuracy"] = dict(
                value=None if good+bad==0 else good/(good+bad), food_count=int(good),
                contact_count=int(good+bad), assigned_lives=2*len(rows), descriptive=True)
    return dict(families=len(rows), arms=arms, contrasts=measures,
                gate=dict(passed=all(criteria.values()), criteria=criteria,
                          failed_criteria=[name for name, passed in criteria.items() if not passed]))


@lru_cache(maxsize=1)
def random_helpers():
    jax, jnp = source_bytes.random_runtime()

    @jax.jit
    def full_script(key):
        kd, kc, ko = jax.random.split(key, 3)
        directions = jax.vmap(lambda k: jax.random.permutation(k, jnp.arange(1, 5)))(jax.random.split(kd, 4))
        first = jax.random.randint(kc, (4, 4), 1, 3)
        colors = jnp.stack((first, 3-first), axis=-1).reshape(-1)
        outward = jnp.repeat(directions, 2, axis=1).reshape(-1)
        offsets = jax.random.randint(ko, (32,), 1, 4)
        other = (outward-1+offsets) % 4+1
        a, b = jnp.where(colors == 1, outward, other), jnp.where(colors == 2, outward, other)
        return jnp.stack((a, b, colors, outward, jnp.asarray([0, 2, 1, 4, 3])[outward]), axis=1).astype(jnp.int32)

    @jax.jit
    def world(key):
        k1, k2 = jax.random.split(key)
        cells = jax.random.choice(k1, 256, (48,), replace=False)
        grid = jnp.zeros(256, jnp.int32).at[cells[:24]].set(1).at[cells[24:]].set(2).reshape(16, 16)
        return grid, jax.random.randint(k2, (1, 2), 0, 16)

    @jax.jit
    def action_tape(key, logits):
        def step(previous, logit):
            following, action_key, regrowth_key = jax.random.split(previous, 3)
            return following, (previous, following, action_key, regrowth_key,
                jax.random.categorical(action_key, logit), jax.random.randint(regrowth_key, (1, 2), 0, 16))
        return jax.lax.scan(step, key, logits)[1]

    return jax, jnp, full_script, world, action_tape


def probe(data, snapshot, target, keys, ordinal=1):
    probabilities = typed(data, "probabilities", (12, 5), "f")
    encounter.softmax(typed(data, "logits", (12, 5), "f"), probabilities)
    exact(data["logits"], np.float32(3)*data["decision_motor"], "native local motor logits")
    require(np.all(np.abs(data["decision_motor"]) <= 1), "bounded local motor state")
    for name in ("hebbian_enabled", "predictor_enabled", "feedback_applied"):
        require(np.shape(data[name]) == (12,) and not np.any(data[name]), "readout has no learning or feedback " + name)
    jax, _, _, _, _ = random_helpers()
    expected_keys = np.asarray(jax.random.split(jax.random.fold_in(keys[5], ordinal), 12))
    exact(data["initial_key"], expected_keys, "paired local probe streams")
    for placement in range(12):
        grid = encounter.pair_grid(PLACEMENTS[placement])
        exact(data["initial_grid"][placement], grid, "rule-blind local placement")
        exact(data["initial_pos"][placement], np.asarray([[8, 8]], np.int32), "neutral local coordinate")
        expected_obs = encounter.observation(grid, np.asarray([8, 8]), 1., 0.)[None]
        exact(data["observation"][placement], expected_obs, "identical complete local input")
    for name, expected in (("energy", 1), ("denergy", 0), ("alive", 1), ("sig", 0), ("t", 0), ("counts", 0)):
        require(np.all(data["initial_"+name] == expected), "neutral local body " + name)
        exact(data["final_"+name], data["initial_"+name], "readout does not move or feed " + name)
    for name in NEURAL:
        exact(data["initial_"+name+"_sha256"], np.broadcast_to(snapshot[name+"_sha256"], (12, 32)), "carry acquired local " + name)
        if name != "x":
            exact(data["final_"+name+"_sha256"], data["initial_"+name+"_sha256"], "frozen readout " + name)
    return float(signed_choice(probabilities, target))


def trace_physics(data, initial, rules, script=None, pre=None, gap=0):
    """Reconstruct imposed exposure or every free grid action from saved events."""
    steps = len(rules)
    required = dict(grid_before=(steps,16,16), grid=(steps,16,16), observation=(steps,1,103),
        action=(steps,1), sampled_action=(steps,1), forced_action=(steps,), position_before=(steps,1,2),
        position=(steps,1,2), logits=(steps,1,5), probabilities=(steps,1,5), decision_motor=(steps,1,5),
        efference=(steps,1,5), explicit_learning_denergy=(steps,1))
    for name, shape in required.items():
        typed(data, name, shape, "iufb")
    exact(data["rule"], np.asarray(rules, np.int32), "physical payoff schedule")
    exact(data["t"], np.arange(steps, dtype=np.int32), "exposure/free clock")
    encounter.softmax(data["logits"], data["probabilities"])
    exact(data["logits"], np.float32(3)*data["decision_motor"], "native five-way motor logits")
    require(np.all(np.abs(data["decision_motor"]) <= 1), "bounded native decision motor")
    require(np.all(data["hebbian_enabled"]) and np.all(data["predictor_enabled"]), "both native learning processes active")
    exact(data["explicit_learning_denergy"], data["incoming_denergy"], "coherent explicit plasticity feedback")
    _, _, _, _, tape = random_helpers()
    random = tape(initial["key"], data["logits"])
    for field, expected in zip(("key_before", "key_after", "action_key", "regrowth_key", "sampled_action", "regrowth_cell"), random):
        exact(data[field], np.asarray(expected), "independently reconstructed " + field)
    if script is None:
        require(np.all(data["forced_action"] == -1), "every scored solo action is voluntary")
        exact(data["action"], data["sampled_action"], "solo uses its own categorical action")
    else:
        actions = np.concatenate((script[:, 3:5].reshape(-1), np.zeros(gap, np.int32)))
        exact(data["forced_action"], actions, "registered exposure script and empty stays")
        exact(data["action"][:,0], actions, "actual imposed replay movement")
    grid, position = initial["grid"].copy(), initial["pos"].copy()
    energy, alive, delta, signal = [initial[name].copy() for name in ("energy", "alive", "denergy", "sig")]
    for tick in range(steps):
        if script is not None:
            if tick < 2*len(script) and tick % 2 == 0:
                grid = encounter.pair_grid(script[tick//2, :2])
            if tick == 2*len(script):
                grid = np.zeros((16,16), np.int32)
        exact(data["grid_before"][tick], grid, "actual resources before action")
        exact(data["position_before"][tick], position, "continuous physical position")
        for field, value in (("energy_before",energy),("alive_before",alive),("incoming_denergy",delta),("signal_before",signal)):
            physical_close(data[field][tick], value, "actual body continuity " + field)
        exact(data["observation"][tick], encounter.observation(grid, position[0], energy[0], delta[0])[None],
              "physical observation includes coherent energy and delta")
        action = int(data["action"][tick,0])
        require(0 <= action < 5, "native action range")
        moved = (position+MOVES[action]*alive[:,None].astype(np.int32)) % 16
        exact(data["position"][tick], moved, "executed action moves the body")
        item = int(grid[tuple(moved[0])])
        food, poison = bool(alive[0] and item == 1+rules[tick]), bool(alive[0] and item == 2-rules[tick])
        require(int(data["item"][tick,0]) == item and bool(data["food"][tick,0]) == food
            and bool(data["poison"][tick,0]) == poison, "observed contacts follow actual mapping")
        raw = energy-np.float32(.003)*alive+np.float32(.25)*food-np.float32(.25)*poison
        updated = np.clip(raw, 0., 2.)
        physical_close(data["unclipped_energy"][tick], raw, "ordinary unclipped energy equation")
        physical_close(data["energy"][tick], updated, "ordinary clipping")
        physical_close(data["denergy"][tick], (data["energy"][tick]-energy)*alive, "felt energy difference")
        exact(data["alive_after"][tick], alive*(data["energy"][tick]>0), "death is absorbing")
        exact(data["efference"][tick], np.eye(5,dtype=np.float32)[[action]], "actual movement reaches feedback")
        if script is not None:
            require(bool(alive[0]) and np.all(raw > .461) and np.all(raw < 1.251), "deterministic exposure bound and survival")
        if food or poison:
            grid[tuple(moved[0])] = 0
            if script is None:
                cell = data["regrowth_cell"][tick,0]
                require(np.all((cell>=0)&(cell<16)), "regrowth coordinate bounds")
                grid[tuple(cell)] = item
        exact(data["grid"][tick], grid, "consumption and declared regrowth")
        position, energy, alive, delta, signal = [data[name][tick] for name in ("position","energy","alive_after","denergy","signal")]
    return encounter.grid_outcomes(data["food"][:,0], data["poison"][:,0], data["alive_after"][:,0])


def state_end(state, trace, stop):
    for field, trace_field in (("grid","grid"),("pos","position"),("energy","energy"),("alive","alive_after"),
                              ("denergy","denergy"),("sig","signal"),("key","key_after")):
        exact(state[field], trace[trace_field][stop-1], "recorded state at boundary " + field)
    require(int(state["t"]) == stop, "state tick matches processed history")
    counts=np.zeros((1,2,2),np.float32)
    for half in (0,1):
        take=(np.arange(stop)>=150)==bool(half)
        counts[0,half]=[np.sum(trace[name][:stop,0][take]) for name in ("food","poison")]
    exact(state["counts"],counts,"contact telemetry is physical and unscored")


def audit_family(raw, index, parent, endpoint, keys, pre, gap, sham):
    require(int(raw["base_index"]) == index and int(raw["pre_contacts"]) == pre and int(raw["gap"]) == gap
            and int(raw["sham_rule"]) == sham, "exact frozen family assignment")
    exact(raw["keys"], keys, "assigned family role keys")
    exact(raw["initial_rules"], np.asarray([0,1],np.int32), "both initial mappings")
    exact(raw["genome_sha256"], np.stack((fingerprint(parent),fingerprint(endpoint))), "exact parent and endpoint")
    _, _, make_script, make_world, _ = random_helpers()
    script = np.asarray(make_script(keys[0]))[:pre+16]
    exact(raw["script"], script, "independent color/direction script")
    grid, position = (np.asarray(value) for value in make_world(keys[3]))
    exact(raw["target_grid"],grid,"ordinary independently generated solo layout")
    exact(raw["target_position"],position,"ordinary solo starting coordinate")
    old, local = np.empty((2,2)), np.empty((2,2,2,2))
    outcome = {name:np.empty((2,2,2)) for name in ("Y","first_food","first_poison","no_contact","survival","latency")}
    groups = {prefix:prefixed(raw,prefix+"_") for prefix in ("birth","boundary","old","acquired","local","exposure","solo_initial","solo_final")}
    solo = {name:value for name,value in prefixed(raw,"solo_").items()
            if not name.startswith(("initial_","final_"))}
    for name,value in groups["birth"].items():
        exact(value[:,0],value[:,1],"both rule labels start from the same inherited state "+name)
    for name,value in groups["exposure"].items():
        for history in (0,1):
            for feedback in (0,1):
                exact(value[:,:,history,feedback,:2*pre],value[:,:,0,0,:2*pre],
                      "shared pre-boundary exposure "+name)
    for genotype in range(2):
        for rule in range(2):
            birth={name:value[genotype,rule] for name,value in groups["birth"].items()}
            boundary={name:value[genotype,rule] for name,value in groups["boundary"].items()}
            require(np.all(birth["energy"]==1) and np.all(birth["alive"]==1) and np.all(birth["denergy"]==0),"fresh body")
            exact(birth["grid"],np.zeros((16,16),np.int32),"fresh empty chamber")
            exact(birth["pos"],np.asarray([[8,8]],np.int32),"fresh chamber hub")
            exact(birth["key"],keys[2],"separate chamber stream")
            old[genotype,rule]=probe({name:value[genotype,rule] for name,value in groups["old"].items()},boundary,rule,keys,0)
            for history in range(2):
                target=rule if history==0 else 1-rule
                for feedback in range(2):
                    address=(genotype,rule,history,feedback)
                    acquired={name:value[address] for name,value in groups["acquired"].items()}
                    trace={name:value[address] for name,value in groups["exposure"].items()}
                    rules=np.full(2*(pre+16)+gap,sham if feedback else target,np.int32)
                    rules[:2*pre]=rule
                    trace_physics(trace,birth,rules,script,pre,gap)
                    state_end(boundary,trace,2*pre)
                    state_end(acquired,trace,len(rules))
                    exact(acquired["mask_sha256"],birth["mask_sha256"],"inherited topology remains fixed")
                    local[genotype,rule,history,feedback]=probe(
                        {name:value[address] for name,value in groups["local"].items()},acquired,target,keys)
                left,right=(genotype,rule,0,1),(genotype,rule,1,1)
                for prefix in ("exposure","acquired","local"):
                    for name,value in groups[prefix].items():
                        exact(value[left],value[right],"yoked histories carry identical input and state " + prefix+"_"+name)
                        matched=(genotype,rule,int(sham!=rule),0)
                        exact(value[left],value[matched],"yoke equals informative branch with same physical mapping " + prefix+"_"+name)
                acquired={name:value[genotype,rule,history,0] for name,value in groups["acquired"].items()}
                initial={name:value[genotype,rule,history] for name,value in groups["solo_initial"].items()}
                final={name:value[genotype,rule,history] for name,value in groups["solo_final"].items()}
                for name in NEURAL:
                    exact(initial[name+"_sha256"],acquired[name+"_sha256"],"solo carries unprobed brain " + name)
                exact(initial["grid"],grid,"matched ordinary solo grid")
                exact(initial["pos"],position,"matched solo coordinate")
                exact(initial["key"],keys[4],"paired solo action stream")
                for name,value in (("energy",1),("alive",1),("denergy",0),("sig",0),("t",0),("counts",0)):
                    require(np.all(initial[name]==value),"neutral solo body " + name)
                trace={name:value[genotype,rule,history] for name,value in solo.items()}
                values=trace_physics(trace,initial,np.full(50,target,np.int32))
                state_end(final,trace,50)
                for name in outcome:
                    outcome[name][genotype,rule,history]=values["food_minus_poison" if name=="Y" else name]
    require(np.array_equal(local[:,:,0,1],-local[:,:,1,1]),
            "yoked same-target history effect is numerically exactly zero before averaging")
    return dict(old=old,local=local,**outcome)


def candidate_scores(samples):
    probabilities=typed(samples,"probabilities",(16,4,2,2,12,5),"f")
    encounter.softmax(typed(samples,"logits",probabilities.shape,"f"),probabilities)
    require(np.all(np.abs(samples["logits"])<=3.),"native training logit range")
    choice=np.empty((16,4,2,2),np.float64)
    for rule in range(2):
        for history in range(2):
            choice[:,:,rule,history]=signed_choice(probabilities[:,:,rule,history],rule if history==0 else 1-rule)
    expected=choice.mean((2,3))
    fitness=typed(samples,"fitness",(16,4),"f")
    physical_close(fitness,expected,"fitness is candidate free-choice expectation")
    # Reproduce the device mean from the saved family values to preserve rank ties.
    _,jnp=source_bytes.random_runtime()
    scores=np.asarray(jnp.asarray(fitness).mean(1))
    return scores


def optimizer_transition(previous,samples):
    scores=candidate_scores(samples)
    eps=typed(samples,"eps",(16,155),"f")
    exact(eps[8:],-eps[:8],"antithetic perturbations")
    jax,jnp=source_bytes.random_runtime()
    next_key,noise_key,world_key,birth_key=jax.random.split(jnp.asarray(previous["key"]),4)
    half=np.asarray(jax.random.normal(noise_key,(8,155)))
    exact(eps[:8],half,"independent mutation noise")
    exact(samples["world_keys"],np.asarray(jax.random.split(world_key,4)),"fresh training family roots")
    births=np.asarray(jax.random.split(birth_key,32)).reshape(8,4,2)
    exact(samples["development_keys"],np.concatenate((births,births)),"paired developmental randomness")
    candidates=np.asarray(jnp.asarray(previous["mu"])+.05*jnp.asarray(eps))
    exact(samples["candidate_genomes"],candidates,"candidate genotype arithmetic")
    # Derive tie positions by sorting, then apply the device's float32 scaling.
    order=np.argsort(scores,kind="stable")
    positions=np.empty(16,np.float32)
    begin=0
    while begin<16:
        end=begin+1
        while end<16 and scores[order[end]]==scores[order[begin]]:
            end+=1
        positions[order[begin:end]]=(begin+end-1)/2
        begin=end
    weights=np.asarray(jnp.asarray(positions)/15-.5)
    exact(samples["rank_weights"],weights,"mean tied ranks")
    informative=bool(np.any(scores!=scores[0]))
    expected={name:previous[name].copy() for name in ("mu","m","v","t")}
    if informative:
        mu,m,v=(jnp.asarray(previous[name]) for name in ("mu","m","v"))
        gradient=(jnp.asarray(weights)[:,None]*jnp.asarray(eps)).sum(0)/(.05*16)-.005*mu
        reference=(weights.astype(np.float64)[:,None]*eps.astype(np.float64)).sum(0)/.8-.005*previous["mu"].astype(np.float64)
        magnitude=(np.abs(weights.astype(np.float64)[:,None]*eps.astype(np.float64))).sum(0)/.8+.005*np.abs(previous["mu"])
        unit=np.finfo(np.float32).eps/2
        require(np.all(np.abs(np.asarray(gradient,np.float64)-reference)<=22*unit/(1-22*unit)*magnitude+22*np.finfo(np.float32).smallest_subnormal),
                "optimizer reduction within independent float64 roundoff bound")
        step=int(previous["t"])+1
        m=.9*m+.1*gradient
        v=.999*v+.001*gradient**2
        mu=mu+.02*(m/(1-.9**step))/(jnp.sqrt(v/(1-.999**step))+1e-8)
        expected.update(mu=np.asarray(mu),m=np.asarray(m),v=np.asarray(v),t=np.asarray(step,previous["t"].dtype))
    expected["key"]=np.asarray(next_key)
    return expected,scores,informative


class Auditor(base.Auditor):
    def __init__(self,root,source):
        require(re.fullmatch(r"[0-9a-f]{40}",source),"full source commit")
        super().__init__(root,source)
        self.training_seconds=[]
        self.family_seconds=[]
        self.source_manifest=None
        self.training_reports={}

    def register_keys(self,keys,label):
        values=np.asarray(keys)
        require(values.dtype==np.uint32 and values.shape[-1:]==(2,),label+" unsigned keys")
        tokens=[value.tobytes() for value in values.reshape(-1,2)]
        require(len(set(tokens))==len(tokens) and not self.used_keys.intersection(tokens),label+" distinct keys")
        self.used_keys.update(tokens)

    def artifact(self,folder,manifest):
        require(set(manifest)=={"path","sha256","bytes"},"complete bounded artifact manifest")
        path=self.path(folder+"/"+manifest["path"])
        require(path.is_relative_to(self.path(folder)),"artifact inside its run")
        content=self.read(path.relative_to(self.root).as_posix())
        require(len(content)==manifest["bytes"] and hashlib.sha256(content).hexdigest()==manifest["sha256"],"saved artifact bytes and SHA")
        require(len(content)<=64<<20,"compressed artifact size bound")
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries=archive.infolist()
            require(0<len(entries)<=2048 and sum(entry.file_size for entry in entries)<=256<<20,"expanded archive bound")
            require(len({entry.filename for entry in entries})==len(entries)
                    and all(entry.filename.endswith(".npy") and "/" not in entry.filename and "\\" not in entry.filename for entry in entries),"unique numeric archive entries")
        with np.load(io.BytesIO(content),allow_pickle=False) as archive:
            arrays={name:archive[name] for name in archive.files}
        require(all(value.dtype.kind in "biuf" and np.isfinite(value).all() for value in arrays.values()),"finite numeric evidence")
        return arrays

    def completed(self,name,attempts,families):
        found=list(self.path("runs").glob("*_"+name))
        require(len(found)<=1,"one registered attempt "+name)
        if not found or not (found[0]/"status.json").is_file():
            raise base.Incomplete(name+" has no terminal status")
        folder=found[0].relative_to(self.root).as_posix()
        status=self.json(folder+"/status.json")
        if status.get("status")!="finished" or status.get("completed_attempts")!=attempts or status.get("completed_families")!=families:
            raise base.Incomplete(name+" incomplete; held-out family arrays remain unopened")
        require(self.path(folder+"/summary.json").is_file(),"terminal summary")
        require({p.name for p in (found[0]/"families").glob("*.npz")}=={f"family_{i:04d}.npz" for i in range(families)},"exact assigned family files")
        return folder

    def metadata(self,folder,index,mode):
        config=self.json(folder+"/config.json")
        development=mode.startswith("development")
        is_training=mode.endswith("training")
        attempts=2 if development and is_training else 10 if is_training else 0
        families=0 if is_training else 4 if development else 128
        training_seed=2026121300+(0 if development else index+1)
        evaluation_seed=2026121400+(0 if development else index+1)
        subset(config,dict(protocol=PROTOCOL,mode=mode,seed_index=index,cfg=CFG,l2=.005,
                           expected_attempts=attempts,expected_families=families),"registered E134 design")
        same(config["training_seed"],training_seed if is_training else None,"registered training split")
        same(config["evaluation_seed"],evaluation_seed if not is_training else None,"registered evaluation split")
        args=config["args"]
        subset(args,dict(eval_protocol=PROTOCOL,generations=attempts,eval_worlds=families,pop=16,worlds=4,
            sigma=.05,lr=.02,seed=training_seed if is_training else evaluation_seed,
            no_plastic=False,allow_dirty=False,quick=False,resume=None,cfg=None,objective=None,memory_condition=None,
            max_train_seconds=None,checkpoint_every=10),"frozen invocation")
        require(args.get("controller") in (None,"rules-v0") and args.get("structural_plasticity") in (None,"off")
                and not args.get("persistence",False),"native controller only")
        commit=config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}",commit)
                and self.code_trees(commit)==self.expected_trees,"clean registered scientific source")
        require(SOURCE_REQUIRED.issubset(config["source_code"]),"required transitive scientific source map")
        context=type("SourceContext",(),{"root":self.root,"source":commit})()
        attributes=source_bytes.Auditor.source_attributes(context,list(config["source_code"]))
        for path,digest in config["source_code"].items():
            blob=base.subprocess.run(["git","show",commit+":"+path],cwd=self.root,capture_output=True,check=True,timeout=15).stdout
            source_bytes.source_hash_representation(blob,digest,attributes[path])
        if self.source_manifest is None:
            self.source_manifest=config["source_code"]
        same(config["source_code"],self.source_manifest,"same development/final source bytes")
        path,digest=base.ORIGINALS[index]
        subset(config["source"],dict(path=path,sha256=digest,training_seed=index),"pinned E103 parent")
        same(args["start"],"file:"+path,"correct original lineage")
        content=self.read(path)
        require(hashlib.sha256(content).hexdigest()==digest and self.read(folder+"/genome_start.npy")==content,"unchanged parent source bytes")
        parent=np.load(io.BytesIO(content),allow_pickle=False)
        require(parent.shape==(155,) and parent.dtype==np.float32 and np.isfinite(parent).all(),"native finite 155-number genome")
        report=self.json(folder+"/summary.json")
        subset(report,dict(status="finished",completed_attempts=attempts,completed_families=families),"complete registered work")
        require(0<report["elapsed_seconds"]<=config["max_seconds"],"bounded process wall time")
        require(len(report["attempt_wall_seconds"])==attempts and len(report["family_wall_seconds"])==families
                and all(np.isfinite(v) and v>=0 for v in report["attempt_wall_seconds"]+report["family_wall_seconds"]),"complete measured stage times")
        return config,report,parent

    def training(self,folder,index,development):
        config,report,parent=self.metadata(folder,index,"development-training" if development else "training")
        attempts=config["expected_attempts"]
        require(len(report["attempts"])==attempts+1 and len(report["samples"])==attempts,"all immutable attempts")
        previous=self.artifact(folder,report["attempts"][0])
        require(set(previous)=={"mu","m","v","t","key"},"optimizer fields")
        exact(previous["mu"],parent,"initial ES parent")
        require(not previous["m"].any() and not previous["v"].any() and int(previous["t"])==0,"fresh optimizer moments")
        jax,_,_,_,_=random_helpers()
        initial=np.asarray(jax.random.split(jax.random.PRNGKey(config["training_seed"]))[0])
        exact(previous["key"],initial,"registered optimizer seed")
        self.register_keys(previous["key"][None],"initial optimizer")
        updates=[]
        for attempt in range(1,attempts+1):
            started=time.monotonic()
            require(report["attempts"][attempt]["path"]==f"generations/attempt_{attempt:04d}.npz"
                    and report["samples"][attempt-1]["path"]==f"generations/samples_{attempt:04d}.npz","ordered attempt artifacts")
            current=self.artifact(folder,report["attempts"][attempt])
            samples=self.artifact(folder,report["samples"][attempt-1])
            expected,scores,informative=optimizer_transition(previous,samples)
            for name,value in expected.items():
                exact(current[name],value,"independent exact optimizer transition "+name)
            self.register_keys(current["key"][None],"optimizer continuation")
            self.register_keys(samples["world_keys"],"fresh training roots")
            self.register_keys(samples["development_keys"][:8],"antithetic births")
            self.training_batch(samples)
            updates.append(dict(attempt=attempt,informative=informative,optimizer_updates=int(current["t"]),
                changed_parameters=int(np.count_nonzero(current["mu"]!=previous["mu"])),
                score_mean=float(scores.mean()),score_best=float(scores.max())))
            previous=current
            self.training_seconds.append(time.monotonic()-started)
        endpoint_bytes=self.read(folder+"/genome_final.npy")
        endpoint=np.load(io.BytesIO(endpoint_bytes),allow_pickle=False)
        exact(endpoint,previous["mu"],"endpoint is last attempted generation")
        same(report["endpoint_sha256"],hashlib.sha256(endpoint_bytes).hexdigest(),"pinned endpoint SHA")
        same(report["optimizer_updates"],int(previous["t"]),"recorded informative update count")
        self.training_reports[folder]=(endpoint,config,report)
        return dict(index=index,run=folder,attempts=updates,endpoint_sha256=report["endpoint_sha256"])

    def training_batch(self,samples):
        batch=prefixed(samples,"batch_")
        exact(batch["world_keys"],samples["world_keys"],"batch shares training roots")
        jax,_,make_script,_,_=random_helpers()
        role_keys=np.stack([np.asarray(jax.random.split(key,3)) for key in samples["world_keys"]])
        exact(batch["role_keys"],role_keys,"independently derived training roles")
        schedule_key=np.asarray(jax.random.fold_in(samples["world_keys"][0],0xE134))
        exact(batch["schedule_key"],schedule_key,"independently derived schedule stream")
        cells=np.asarray([(8,0),(8,8),(16,0),(16,8)],np.int32)[np.asarray(jax.random.permutation(schedule_key,4))]
        exact(batch["pre_contacts"],cells[:,0],"balanced randomized pre contacts")
        exact(batch["gap"],cells[:,1],"balanced randomized training gaps")
        self.register_keys(batch["role_keys"],"fresh training script/chamber/probe roles")
        self.register_keys(schedule_key[None],"fresh training schedule")
        for family in range(4):
            expected_script=np.asarray(make_script(role_keys[family,0])).copy()
            expected_script[int(cells[family,0])+16:]=0
            exact(batch["scripts"][family],expected_script,"training script independently reconstructed with zero padding")
        counts=typed(samples,"counts",(16,4,2,2),"fiu")
        for family,pre in enumerate(batch["pre_contacts"]):
            expected=np.broadcast_to([[int(pre)//2,int(pre)//2],[8,8]],(16,2,2))
            physical_close(counts[:,family],expected,"truthful prescribed exposure counts")
        require(np.shape(samples["survived"])==(16,4) and np.all(samples["survived"]==1),"all assigned exposure bodies live")
        bounds=typed(samples,"exposure_energy_bounds",(16,4,2),"f")
        require(np.all(bounds[...,0]>=.462-1e-5) and np.all(bounds[...,1]<=1.25+1e-5)
                and np.all(bounds[...,0]<=bounds[...,1]),"guaranteed unclipped training energy range")

    def evaluation(self,folder,index,development):
        config,report,parent=self.metadata(folder,index,"development-evaluation" if development else "evaluation")
        receipts=config["train_receipts"]
        require(len(receipts)==(1 if development else 3),"all endpoints pinned before evaluation")
        endpoint=None
        for lineage,receipt in enumerate(receipts):
            require(receipt["seed_index"]==lineage and receipt["run"] in self.training_reports,"endpoint from complete training")
            trained,_,training_report=self.training_reports[receipt["run"]]
            names=dict(config_sha256="config.json",summary_sha256="summary.json",
                checkpoint_sha256=f"generations/attempt_{2 if development else 10:04d}.npz",endpoint_sha256="genome_final.npy")
            for field,path in names.items():
                same(receipt[field],hashlib.sha256(self.read(receipt["run"]+"/"+path)).hexdigest(),"frozen training receipt "+field)
            if lineage==index:
                endpoint=trained
                same(config["endpoint_sha256"],receipt["endpoint_sha256"],"selected own endpoint")
        content=self.read(folder+"/genome_final.npy")
        same(hashlib.sha256(content).hexdigest(),config["endpoint_sha256"],"evaluation frozen endpoint copy")
        exact(np.load(io.BytesIO(content),allow_pickle=False),endpoint,"evaluation descendant identity")
        assignment=self.artifact(folder,config["assignments"])
        count=config["expected_families"]
        jax,_,_,_,_=random_helpers()
        base_key=jax.random.fold_in(jax.random.PRNGKey(config["evaluation_seed"]),0xE134)
        manifest_key,family_key=jax.random.split(base_key)
        table=np.asarray([(8,16,0),(16,32,0),(8,32,1),(16,16,1)],np.int32) if development else np.repeat(
            np.asarray([(pre,gap,sham) for pre in (8,16) for gap in (16,32) for sham in (0,1)],np.int32),16,axis=0)
        if not development:
            table=table[np.asarray(jax.random.permutation(manifest_key,count))]
        family_keys=np.stack([np.asarray(jax.random.split(jax.random.fold_in(family_key,i),6)) for i in range(count)])
        for column,name in enumerate(("pre_contacts","gap","sham_rule")):
            exact(assignment[name],table[:,column],"independent balanced final "+name)
        exact(assignment["family_keys"],family_keys,"new held-out family role streams")
        exact(assignment["initial_rules"],np.asarray([0,1],np.int32),"both rule replays assigned")
        self.register_keys(family_keys,"held-out scripts/births/probes/grids")
        require(len(report["raw_files"])==count,"all family artifacts")
        rows=[]
        for family,manifest in enumerate(report["raw_files"]):
            started=time.monotonic()
            require(manifest["path"]==f"families/family_{family:04d}.npz","ordered independent family")
            raw=self.artifact(folder,manifest)
            rows.append(audit_family(raw,family,parent,endpoint,family_keys[family],*map(int,table[family])))
            self.family_seconds.append(time.monotonic()-started)
        return dict(index=index,run=folder,measurements=summary(rows))


def self_check():
    uniform=np.full((12,5),.2,np.float32)
    exact(signed_choice(uniform,0),np.asarray(0.),"uniform five-way policy scores zero")
    good=np.zeros((12,5),np.float32)
    good[np.arange(12),PLACEMENTS[:,0]]=1
    require(signed_choice(good,0)==1 and signed_choice(good,1)==-1,"signed directions and all placements")
    require(signed_choice(good,0)-signed_choice(good,1)==2,"perfect opposing policies have history effect two")
    require(signed_choice(good,0)-signed_choice(good,0)==0,"identical policy has zero same-target history effect")
    rows=[]
    for i in range(4):
        old=np.full((2,2),.3)
        local=np.zeros((2,2,2,2))
        local[:,:,0,0]=.3
        local[0,:,1,0]=.1
        local[1,:,1,0]=.3
        local[:,:,:,1]=0
        y=np.zeros((2,2,2))
        y[:,:,0]=.3
        y[0,:,1]=.1
        y[1,:,1]=.3
        rows.append(dict(old=old,local=local,Y=y,first_food=(y+1)/2,first_poison=(1-y)/2,
                         no_contact=np.zeros_like(y),survival=np.ones_like(y),latency=np.ones_like(y)))
    result=summary(rows)
    require(result["gate"]["passed"],"known positive paired fixture passes")
    changed=[dict(row,local=row["local"].copy()) for row in rows]
    for row in changed:
        row["local"][1,:,1,0]=row["local"][0,:,1,0]
    require(not summary(changed)["gate"]["passed"],"direct descendant improvement is required")
    spread=statistics([-.5,.5,-.5,.5])
    require(spread["se"]>0 and statistics([0.,0.,0.,0.])["se"]==0,"paired contrast has its own family SE")
    exact(average_ranks(np.asarray([1,1,2,2,3],np.float32)),np.asarray([-.375,-.375,.125,.125,.5],np.float32),"mean tied ranks")
    return dict(status="self-check passed",groups=7,simulator_imported=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--development",action="store_true")
    parser.add_argument("--training",action="store_true")
    parser.add_argument("--self-check",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(),allow_nan=False))
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required")
    started=time.monotonic()
    auditor=None
    report=dict(protocol=PROTOCOL,mode="development" if args.development else "training" if args.training else "final")
    try:
        auditor=Auditor(args.root,args.source_commit)
        dev_train=auditor.completed("E134-benchmark",2,0)
        dev_eval=auditor.completed("E134-benchmark-eval",0,4)
        final_train=[] if args.development else [auditor.completed(f"E134-A{i}-train",10,0) for i in range(3)]
        final_eval=[] if args.development or args.training else [auditor.completed(f"E134-A{i}",0,128) for i in range(3)]
        report["training"]=[auditor.training(dev_train,0,True)]
        report["development"]=auditor.evaluation(dev_eval,0,True)
        report["training"].extend(auditor.training(folder,i,False) for i,folder in enumerate(final_train))
        report["runs"]=[auditor.evaluation(folder,i,False) for i,folder in enumerate(final_eval)]
        report.update(status="passed",registered_source=args.source_commit,registered_source_code=auditor.source_manifest,
            final_evaluation_audited=bool(final_eval),scientific_gate_passed=all(item["measurements"]["gate"]["passed"] for item in report["runs"]) if final_eval else None,
            limitations=["Neural matrix operations and candidate development are not independently rerun.",
                         "Physics, PRNG assignments/actions, five-way choice scores, ES updates and paired-family decisions are independently reconstructed.",
                         "Rule-uninformative replay is a local information control; solo release uses informative exposure."])
        code=0
    except base.Incomplete as error:
        report.update(status="incomplete",error=str(error),final_evaluation_audited=False)
        code=2
    except (base.AuditError,OSError,ValueError,KeyError,TypeError,base.subprocess.SubprocessError) as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}",final_evaluation_audited=False)
        code=1
    elapsed=time.monotonic()-started
    if auditor is not None:
        report["checked_file_sha256"]=auditor.hashes
        report["timing"]=dict(training_attempt_wall_seconds=auditor.training_seconds,family_wall_seconds=auditor.family_seconds,
            elapsed_seconds=elapsed,setup_and_summary_seconds=max(0.,elapsed-sum(auditor.training_seconds)-sum(auditor.family_seconds)))
    encoded=json.dumps(report,indent=2,allow_nan=False)+"\n"
    if args.output is not None:
        with args.output.open("x",encoding="utf-8",newline="\n") as stream:
            stream.write(encoded)
        print(json.dumps(dict(status=report["status"],output=str(args.output),scientific_gate_passed=report.get("scientific_gate_passed"),
                              error=report.get("error")),allow_nan=False))
    else:
        print(encoded,end="")
    return code


if __name__=="__main__":
    sys.exit(main())
