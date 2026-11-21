"""Builds Seed_v0.ipynb from seed.py plus explanation and experiment cells."""
import json, sys
from pathlib import Path

seed_src = open(Path(__file__).resolve().parents[1] / "seed" / "seed.py").read()
structural_src = (Path(__file__).resolve().parents[1] / "seed" / "structural.py").read_text()
seed_src = seed_src.replace("from seed.structural import INTERVAL, STREAM_TAG, birth_caps, rewire, validate_structure", structural_src)
ecology_src = (Path(__file__).resolve().parents[1] / "seed" / "ecology.py").read_text()
seed_src = seed_src.replace("from seed.ecology import EcologyV1, apply_costs, make_shelter", ecology_src)
persistence_src = (Path(__file__).resolve().parents[1] / "seed" / "persistence.py").read_text()
persistence_src = persistence_src.replace("from seed.ecology import EcologyV1", "")
seed_src = seed_src.replace("from seed.persistence import PersistenceV1, bias_logits, initial_memory, validate_persistence", persistence_src)
QUICK = "--quick" in sys.argv

cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))

md(r"""
# Seed v0: brains that grow, learn and evolve on their own

**What this is.** A first experiment in an AI architecture that is never trained with labels. There are three loops:

1. **Evolution (generations).** A population of tiny genomes (155 numbers each) improves over generations.
2. **Development (birth).** Each genome *grows* a brain from wiring rules. It holds rules, not weights, and siblings come out slightly different.
3. **Learning (lifetime).** There's no teacher and no backprop. Synapses change by a local Hebbian rule, gated by a "that mattered" neuromodulator that reacts to felt energy change and to surprise. A local prediction loop learns to guess the next sensation.

**The world.** A small 2D world with two kinds of blobs, A and B. In every new world, a hidden rule decides which is food and which is poison. No inborn instinct can beat chance, so **the only way to win is to learn during life.** Agents live in families of four relatives, see each other, and can emit a signal the others can sense. Energy runs out and agents die.

**Status.** This is an exploratory notebook with historical examples, not the current registered research workflow. Follow `PLAN.md` and run tracked experiments through `scripts/run_experiment.py`. The owner chose laptop-only research; the proposed GPU benchmark was cancelled before execution. Notebook runtime has not been established for the current environment. `QUICK = True` selects a smaller exploratory workload, not a timed guarantee.

**What we're testing.** Does evolution discover brains that *learn*? We check this two ways:
- **Accuracy rises within a life.** If food-choice accuracy in the second half of life is higher than in the first, the agents learned *this* world's rule.
- **The lesion test.** Freeze all synapses in an evolved brain. If performance collapses, plasticity was doing real work.
""")

code(r"""
import jax
print("JAX", jax.__version__, "| devices:", jax.devices())
if jax.devices()[0].platform == "cpu":
    print("CPU backend. Check available memory and benchmark before a large exploratory workload.")
""")

md("## 1. The Seed engine\nThe whole architecture is in this one cell: world, development, brain, lifetime and evolution. Read the comments, then change things.")
code(seed_src)

md(r"""
## 2. Settings
`POP` is the number of genomes per generation, `WORLDS` the number of worlds each genome is tested in (half with each food rule), and `GENERATIONS` how long evolution runs.
""")
code(f"""
QUICK = {QUICK}          # True = 2-minute smoke test

c = Cfg()                # world/brain settings live in Cfg (see cell above)
if QUICK:
    POP, WORLDS, GENERATIONS = 16, 2, 5
else:
    POP, WORLDS, GENERATIONS = 128, 8, 200
print("genome size:", genome_size(c), "numbers | neurons per brain:", dims(c)[3])
""")

md(r"""
## 3. Is learning even possible here? (a 1-minute check)
Before evolving anything, we compare three brains in 32 unseen worlds:
- a **random genome**, which should score near chance;
- a **biology-seeded genome**, using the rule nature already found: dopamine-gated Hebbian learning, where links from senses to movement strengthen after an action that raised energy and weaken after one that lowered it;
- the **same biology-seeded brain, frozen**, with no plasticity.

In our test run, the biology-seeded brain's food-choice accuracy rose from about 0.58 to 0.69 within a single life, and freezing it dropped fitness from about 6.3 to 1.1. So the architecture *can* learn. The open question is whether evolution can *find* this rule, and improve on it.
""")
code(r"""
flat_random, unravel = ravel_pytree(init_genome(c, jax.random.PRNGKey(0)))
flat_bio, _ = ravel_pytree(biology_prior(c))
def report(name, r):
    print(f"{name:22s} | fitness {r['fitness']:5.2f} | accuracy 1st half {r['acc_first']:.2f} -> 2nd half {r['acc_second']:.2f} | survived {r['survived']:.2f}")
report("random genome", test_genome(flat_random, c, n_worlds=32))
report("biology-seeded", test_genome(flat_bio, c, n_worlds=32))
report("biology-seeded, frozen", test_genome(flat_bio, c, n_worlds=32, plastic=False))
""")

md(r"""
## 4. Experiment A: can evolution discover learning from a blank start?
This is the hard, open question. In our small CPU test (32 genomes, 50 generations) it did **not**: fitness stayed near chance. This run is 4× bigger in both population and worlds, and 6× longer. Watch the accuracy columns. If the 2nd-half number climbs above the 1st-half number, evolution has invented learning.
""")
code(r"""
import time, numpy as np
t0 = time.time()
mu_A, unravel, hist_A = evolve(c, generations=GENERATIONS, pop=POP, E=WORLDS, log_every=10)
print(f"done in {(time.time()-t0)/60:.1f} min")
np.save("seed_genome_A.npy", np.asarray(mu_A))
""")

md(r"""
## 5. Experiment B: can evolution improve on biology?
This run starts from the biology-seeded genome and lets evolution change any of its 155 numbers. Mutations are kept small here (`sigma=0.05`, `lr=0.02`). With bigger steps, our test showed evolution scrambling the prior faster than it could improve it. With these settings, a small CPU test held steady (fitness 6.3 → 6.3 over 20 generations). The question for your GPU run is whether it climbs.
""")
code(r"""
t0 = time.time()
mu_B, _, hist_B = evolve(c, generations=GENERATIONS, pop=POP, E=WORLDS, log_every=10, mean=flat_bio, sigma=0.05, lr=0.02)
print(f"done in {(time.time()-t0)/60:.1f} min")
np.save("seed_genome_B.npy", np.asarray(mu_B))
""")

md("## 6. Results")
code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for name, h in [("A: blank start", hist_A), ("B: biology start", hist_B)]:
    g = [r["gen"] for r in h]
    ax[0].plot(g, [r["fit_mean"] for r in h], label=name)
    ax[1].plot(g, [r["acc_second"] - r["acc_first"] for r in h], label=name)
ax[0].set(title="Population fitness (food minus poison per agent)", xlabel="generation"); ax[0].legend()
ax[1].axhline(0, ls="--", c="gray", lw=1)
ax[1].set(title="Within-life improvement in accuracy\n(2nd half minus 1st half; > 0 = learning)", xlabel="generation"); ax[1].legend()
plt.tight_layout(); plt.show()
""")

md(r"""
## 7. The lesion test, in unseen worlds
Each evolved genome lives in new worlds twice: once normally, once with every synapse frozen. If freezing makes it much worse, the brain depends on learning during its life.
""")
code(r"""
N = 16 if QUICK else 128
for name, mu in [("A: blank start", mu_A), ("B: biology start", mu_B), ("biology prior (no evolution)", flat_bio)]:
    print(name)
    report("   learning", test_genome(mu, c, n_worlds=N))
    report("   frozen", test_genome(mu, c, n_worlds=N, plastic=False))
""")

md(r"""
## 8. Watch a family live
Green = food and red = poison **in this world** (the agents only see "A" and "B" and must work out which is which). Circles are agents, coloured by the signal they're emitting (blue = −1, red = +1). An × marks an agent that has died.
""")
code(r"""
from matplotlib import animation
from IPython.display import HTML

def show_life(mu, rule=0, seed=7, step=2):
    g = unravel(mu)
    r = rollout(g, jax.random.PRNGKey(seed), jax.random.PRNGKey(seed + 1), rule, c, True, True)
    tr = jax.tree_util.tree_map(np.asarray, r["traj"])
    food = 1 + rule
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    def draw(t):
        ax.clear()
        grid = tr["grid"][t]
        img = np.ones((c.G, c.G, 3))
        img[grid == food] = [0.35, 0.75, 0.35]
        img[grid == 3 - food] = [0.85, 0.35, 0.35]
        ax.imshow(img)
        for i in range(c.M):
            y, x = tr["pos"][t, i]
            if tr["alive"][t, i]:
                ax.scatter(x, y, s=140, c=[tr["sig"][t, i]], cmap="coolwarm", vmin=-1, vmax=1, edgecolors="k")
            else:
                ax.scatter(x, y, s=80, marker="x", c="k")
        ax.set_title(f"step {t}   energy: " + " ".join(f"{e:.1f}" for e in tr["energy"][t]), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    anim = animation.FuncAnimation(fig, draw, frames=range(0, c.T, step), interval=90)
    plt.close(fig)
    print("counts (1st half, 2nd half) x (food, poison):\n", np.asarray(r["counts"]))
    return HTML(anim.to_jshtml())

show_life(mu_B, rule=0)   # try mu_A, flat_bio, or rule=1
""")

md(r"""
## 9. Where to take it next
Each of these is a single, testable change:
- **Seed from the fly.** Set the starting `conn` (wiring between type pairs) from FlyWire or MaleCNS connectome statistics instead of a flat −1, and compare with random starts.
- **Sleep.** Every N steps, stop acting and replay recent observations through the prediction loop, then test whether agents forget less when worlds change mid-life.
- **Eligibility traces.** Add a genome-controlled fading memory of recent activity, so rewards that arrive a few steps late can still change the right synapses.
- **Meaningful signals.** Measure whether an agent's signal predicts food nearby. If it does, communication has emerged.
- **Strangers, not family.** Fill each world with agents from different genomes and see whether cooperation survives.
""")

nb = dict(cells=cells, metadata=dict(
    accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
    kernelspec=dict(display_name="Python 3", name="python3"),
    language_info=dict(name="python")), nbformat=4, nbformat_minor=0)
out = str(Path(__file__).resolve().parents[1] / "notebooks" / ("Seed_v0_quick.ipynb" if QUICK else "Seed_v0.ipynb"))
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
