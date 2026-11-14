# Research map: practical next steps for Seed

Reviewed 14 November 2026. This is a targeted primary-source review, not an exhaustive novelty search. The experiments below are proposals unless explicitly identified as already approved. Cost labels describe evaluation versus training work; they are not measured runtime estimates.

**Owner direction, 14 November:** develop activity-dependent growth and pruning of connections between existing neurons. This adds lifetime structural plasticity beyond Seed's current fixed connection mask. It is a requested capability, not a demonstrated benefit; implementation, compatibility checks, and scientific comparisons must be recorded separately. LNDP below is directly relevant prior art for this work. No claim of reproducing human neurobiology follows.

## What the literature changes

Evolving local, reward-modulated learning in embodied agents is established prior art. Seed's replicated A result is a project milestone. A thesis contribution needs a precise question, appropriate comparisons, and evidence beyond the general claim that evolution can discover plasticity.

The immediate opportunity is to explain the current result and test its limits using saved genomes before adding mechanisms. Keep the current C comparison, approved predictor diagnostic, and planned B replications traceable; preserve EVAL v1 throughout.

## Closest prior work

Giannakakis, Khajehabdollahi and Levina's [2024 preprint, *Network bottlenecks and task structure control the evolution of interpretable learning rules in a foraging agent*](https://arxiv.org/html/2403.13649v1), evolves reward-modulated plasticity in embodied foraging. Its methods jointly optimize motor weights and an eight-term sensory plasticity rule with an elitist genetic algorithm. A curriculum progressively moves initial sensory weights away from true ingredient values. The paper examines bottlenecks, regularization, normalization, and sensory/motor module swaps.

| Aspect | That paper | Current Seed |
|---|---|---|
| World | Continuous space; foods contain valued ingredients | Discrete grid; two food/poison types |
| Controller | Plastic sensory network, nonplastic motor network | Recurrent brain grown from type-pair connection rules |
| Search | Genetic algorithm and curriculum | Antithetic evolution strategies; A uses blank initialization |
| Learning | Eight-term rule and normalization | Four-term rules, evolved modulation/rates, clipping, predictor |

These are implementation differences, not proof of novelty or superiority. Fitness numbers across these tasks are not directly comparable. Seed implementation: [genome and development](../seed/seed.py), [experiment evidence](../experiments/LOG.md).

## Reading-to-experiment map

| Priority | Primary publication | Useful idea | Smallest relevant Seed test | Cost / dependency |
|---|---|---|---|---|
| Next | [Giannakakis et al., 2024](https://arxiv.org/html/2403.13649v1) | Identify which plasticity terms matter | Ablate selected rule terms in saved A genomes; retain originals and hash each derived genome | Evaluation panel; new preregistration |
| Next | [Giannakakis et al., 2023, *Environmental variability and network structure determine the optimal plasticity mechanisms in embodied agents*](https://arxiv.org/html/2303.06734v1) | Learning rules depend on environmental changes and feedback reliability | Separate food-rule reversal and noisy-observation conditions; compare learning against frozen | New world/measurement code and versioned protocol |
| Next | [Cobbe et al., 2020, *Leveraging Procedural Generation to Benchmark Reinforcement Learning*](https://proceedings.mlr.press/v119/cobbe20a.html) | Distinguish unfamiliar instances from broader generalization | Separate development conditions from untouched evaluation conditions; test all three saved A genomes | Evaluation panel; no Procgen installation required |
| After predictor diagnostic | [Soltoggio et al., 2008, *Evolutionary Advantages of Neuromodulated Plasticity in Dynamic, Reward-based Scenarios*](https://andrea.soltoggio.net/data/papers/SoltoggioALife2008.pdf) | Isolate neuromodulation rather than lesioning all learning | Remove the modulator's energy-change coefficient while retaining ordinary energy observations | Evaluation panel; separate intervention, not already approved |
| Later, if delayed-credit failure appears | [Miconi et al., *Backpropamine*, ICLR 2019; cited arXiv version 2020](https://arxiv.org/html/2002.10585v1) | Decaying eligibility traces preserve recent activity for later modulation | Introduce one trace mechanism; compare matched new training against the current update | New state and training; benchmark first |
| Later, after transfer measurement works | [Wang et al., 2020, *Enhanced POET*](https://proceedings.mlr.press/v119/wang20l.html) | Generate challenges alongside agents and transfer solutions | Start with a bounded challenge generator and independently held-out tests | New experiment system; substantial additional compute |

The 2023 foraging paper explicitly studies changing food values and unreliable feedback, so adding those conditions alone would not establish novelty. Procgen supplies a useful evaluation discipline; its full benchmark has a different interface and is not a drop-in test for Seed. Soltoggio's evolved modulatory controllers already demonstrate this mechanism's value in T-maze tasks. Backpropamine trains its plastic mechanisms with gradients, unlike Seed's ES; the transferable idea is a trace mechanism, not evidence that it will improve Seed. POET remains constrained by its environment representation; it does not remove the need for designed rules or independent evaluation.

Verified author repositories: [Backpropamine](https://github.com/uber-research/backpropamine), [Procgen](https://github.com/openai/procgen). No code repository was verified for the closest foraging paper. The table cites the actual preprint versions reviewed rather than assuming a later journal version has identical methods.

## Make each proposed test capable of failing

1. **Mechanism interventions:** evaluate all three A genomes with matched worlds and developmental randomness. Report signed fitness changes, including zero or harmful effects. A small effect weakens a claim of necessity; it does not prove the mechanism is never useful. Keep derived genomes separate from original run artifacts. Preregister intervention selection and any claim threshold before evaluation; label later exploratory tests.
2. **Broader conditions:** hold inherited genomes fixed while permitting lifetime learning. Test one shift at a time, then combinations separately. Report food minus poison, survival, and adaptation after a change; pooled accuracy alone can be misleading. Preserve failures. Any tuning uses development conditions, followed by fresh untouched evaluation conditions.
3. **Delayed credit:** establish the baseline failure before adding traces. Check that energy observations or another channel do not reveal the supposedly delayed feedback immediately. Existing genomes were not optimized for a new mechanism; a fair benefit comparison needs matched training budgets and three seeds.
4. **Open-ended claims:** distinguish new layouts, parameter shifts, changing rules, new task families, and persistent challenge generation. Demonstrating one level does not establish all the others.

Any new environment or evaluation extension requires its own preregistration, version, reference evaluations, and the owner's approval under [AGENTS.md](../AGENTS.md). This review authorizes no implementation or run. The already approved EVAL v2 diagnostic remains a separate, narrower comparison of Both, Hebbian-only, and Frozen conditions.

## Candidate thesis framing

An evidence-driven question is: **under a compact developmental encoding, which local mechanisms explain evolved adaptation, and how does that adaptation survive specified environmental changes?** This is a candidate framing, not an established original contribution. Establish the exact prior-art gap before claiming novelty, general intelligence, or superiority to other systems.

## What would justify a state-of-the-art claim?

The owner's goal is to advance the field. Current A/B/C experiments establish internal comparisons, not competitive superiority. A credible competitive claim needs all of the following:

- A named target: the exact task family, adaptation setting, metric, and resource budget being improved. Avoid an unrestricted claim about intelligence.
- Relevant strong published baselines, including the closest plasticity methods and an appropriate conventional controller. Select and justify these before observing the final comparison; a weak baseline is insufficient.
- The same observation/action interface, training distribution, untouched evaluation conditions, and clearly reported training/evaluation interaction and compute budgets. Report parameter counts and encoding differences; matching only one budget does not make every other comparison equivalent.
- Repeated independent training trials, all individual results, and uncertainty appropriate to the sampling unit. The current three-seed minimum is not automatically enough to resolve a small competitive advantage; repeated evaluations of one trained genome are not extra training seeds.
- Baseline implementations checked against their intended behavior, reproducible commands/commits, preserved negative results, and explicit limitations. If a published method cannot be reproduced or aligned, mark that comparison unavailable rather than claiming a win.

A useful first target is **adaptation quality and cost under a preregistered environmental shift**, rather than a higher score on the current task alone. This target remains a proposal. If Seed loses, the comparison should reveal the missing capability and guide the next experiment. Literature review and credible comparisons continue alongside the authorized experiment queue; this document does not substitute a new benchmark for existing commitments.

## Competitive comparison candidates

| Method | Why it is relevant | Reproduction and fairness requirements |
|---|---|---|
| [Lifelong Neural Developmental Programs (LNDP), Plantec et al., 2024, *Evolving Self-Assembling Neural Networks*](https://arxiv.org/html/2406.09787v1) | Combines a developmental encoding with lifetime changes to weights and connectivity, using a graph-transformer implementation. This is close prior art for developmental adaptation, including structural plasticity beyond Seed's fixed lifetime connection mask. | The paper uses CMA-ES for 10,000 generations, populations 128/256, and five seeds. Its switching-food task has five positions and three actions, unlike Seed's world. [Author code](https://github.com/erwanplantec/LNDP) is available under MIT licensing. CPU feasibility and reproduction have not been tested here. |
| [NEAT-NEOL, Lin and Chen, IJCAI 2026, *Provably Sub-Linear Two-Timescale NeuroEvolution with Online Plasticity*](https://ijcai.org/proceedings/2026/703) | Combines architecture search and online reward-modulated plasticity. [Author code](https://github.com/boobaa2001/NeuroEvolution_Online_Learning_NEOL) exposes Hebb, Oja, and BCM alternatives and conventional RL comparisons. | Audit the exact paper configuration and baseline code first. Current README defaults use discrete LunarLander for NEOL but continuous LunarLander for RL, different Hopper versions, and optional inheritance of plasticity-updated weights. These choices must be aligned or explicitly reported. Availability of code does not establish a faithful reproduction. |

The first feasible comparison could place fixed Hebb/Oja/BCM alternatives in a common foraging interface, keeping optimizer and reset conventions controlled. This would be an **adapted local-rule baseline**, not a reproduction of the entire NEAT-NEOL algorithm. A full method comparison requires its topology search as well. LNDP is a stronger developmental comparison candidate with a larger reproduction effort.

The proposed target is higher held-out adaptation performance, or fewer total training interactions to reach a fixed preregistered level, under matched task and resource conditions. Beating candidates on our custom world supports a specific comparative result. A broader state-of-the-art claim additionally requires competitive performance under established external protocols and a justified, current baseline set. None of these comparisons has been run.
