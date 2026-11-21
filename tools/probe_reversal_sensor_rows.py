r"""Bounded development-only falsification of a sensor-row microtick optimization.

No rollout, training, held-out source, checkpoint or run folder is read here.
The dimensions are the production G16/M1/T300/R2/H48, batch128, float32.
This probe neither times full ES nor permits deployment. It requires the released
60-second guard, clean committed source manifest and unchanged CPU environment.

Registered PowerShell payload (the external guard supplies binding variables):
  & .venv\Scripts\python.exe tools/probe_reversal_sensor_rows.py --stage parity --output-dir .tmp/E126-row-parity-01

Inspect compiles only the old kernels from abstract shapes, without fixture
execution. Parity first performs that inspection, then compares one full neural
tick (two recurrent microsteps) on nontrivial synthetic state. If both old kernels
unambiguously already prune the rows, it stops without compiling a candidate.
Neither a byte match nor a compiler finding is a scientific result or permission
to deploy. Full-generation parity, synchronized timing and an immutable source
transition remain separate requirements. No production or evaluation run is read.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BATCH = 128
FIXTURE_SEED = 0xE126D00
REPLACEMENT = "x = jnp.concatenate((obs, jnp.tanh(W[D:, :] @ x)))"
ORIGINAL = "x = jnp.tanh(W @ x).at[:D].set(obs)"
PINNED_SOURCE = {
    "seed/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "seed/direct.py": "a4ede166426bb5fb9abebce4cabc22fc4bd983457d8c516aee3471b66f5ee265",
    "seed/ecology.py": "46281384944377d60404b087e04c9ca6620e912b9c1ab8268438e0a171f439a6",
    "seed/lineage.py": "692f0c534ffd1513c4db4400315c2e91b7469ba19eff19a33df193619952a7e4",
    "seed/paired.py": "361e276d139a82fd05e072a5a97df3cbda74e19bf2840804743f511f9dc653d2",
    "seed/persistence.py": "a20034d0997f6a5555bedd357562566973ed56ac6f8c9b0424efda4a9cf4be23",
    "seed/seed.py": "a202aa0b75abb11e0b0f3dc9d45b85a9ebf8d308fffbb9dfe5b0e268acc41360",
    "seed/structural.py": "56b89b2d82d2eb978840e2565763c254ab28623efa9c339f6b4f4c621c140630",
    "requirements.txt": "1b9c7c01b6356bf0e3bddffe96da53b46c0a952a11a99d0bd44fe3bb5436eb94",
}
PINNED_RUNTIME = {"python": "3.12.10", "numpy": "2.5.4", "jax": "0.12.0.1", "jaxlib": "0.12.0.1"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_text(path, content):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def scratch_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to((ROOT / ".tmp").resolve()) or path == (ROOT / ".tmp").resolve():
        raise ValueError("guard bindings and output must stay under this repository's .tmp on D:")
    return path


def verify_source_manifest(path, expected_sha):
    content = path.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or sha(content) != expected_sha:
        raise ValueError("guard source manifest fingerprint differs")
    raw = json.loads(content.decode("utf-8-sig"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("guard source manifest must be a nonempty path-to-SHA256 map")
    manifest = {}
    for relative, expected in raw.items():
        relative = relative.replace("\\", "/")
        path = (ROOT / relative).resolve()
        if (relative in manifest or not path.is_relative_to(ROOT) or path == ROOT
                or relative.split("/")[0] not in ("seed", "scripts", "tests", "tools", "requirements.txt")
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected.lower())):
            raise ValueError("invalid or duplicate source manifest path/hash")
        if sha(path.read_bytes()) != expected.lower():
            raise ValueError("source differs from guard manifest: " + relative)
        manifest[relative] = expected.lower()
    if any(manifest.get(path) != expected for path, expected in PINNED_SOURCE.items()):
        raise ValueError("production source or requirements differ from the reviewed microtick reference")
    if manifest.get(Path(__file__).resolve().relative_to(ROOT).as_posix()) != sha(Path(__file__).read_bytes()):
        raise ValueError("guard manifest does not bind this exact probe")
    return manifest


def verify_guard(args):
    expected_commit = os.environ.get("SEED_PROBE_EXPECTED_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("the released external guard must bind a committed source identity")
    manifest_path = scratch_path(os.environ.get("SEED_PROBE_SOURCE_MANIFEST", ""))
    manifest_sha = os.environ.get("SEED_PROBE_SOURCE_MANIFEST_SHA256", "")
    manifest = verify_source_manifest(manifest_path, manifest_sha)
    guard_path = scratch_path(os.environ.get("SEED_PROBE_GUARD_RECORD", ""))
    content = guard_path.read_bytes()
    records = [json.loads(line) for line in content.decode("utf-8-sig").splitlines() if line]
    invoked = [row for row in records if row.get("event") == "invoked"]
    preflight = [row for row in records if row.get("event") == "preflight"]
    if (len(invoked) != 1 or len(preflight) != 1 or invoked[0].get("study") != "E126"
            or invoked[0].get("hardLimitSeconds") != 60 or preflight[0].get("commit") != expected_commit
            or invoked[0].get("expectedCommit") != expected_commit
            or any(row.get("event") in ("stopped", "finished", "child-finished") for row in records)):
        raise ValueError("guard must have a live, matching E126 preflight with the fixed 60-second cap")
    arguments = [str(value).replace("\\", "/") for value in preflight[0].get("arguments", [])]
    suffix = ["tools/probe_reversal_sensor_rows.py", "--stage", args.stage, "--output-dir",
              args.output_dir.replace("\\", "/")]
    if len(arguments) < len(suffix) or arguments[-len(suffix):] != suffix:
        raise ValueError("guard command does not bind this exact probe stage and output")
    if (os.environ.get("JAX_PLATFORMS") != "cpu"
            or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false"
            or os.environ.get("PYTHONUTF8") != "1"
            or any(Path(os.environ.get(name, "")).resolve() != (ROOT / ".tmp").resolve() for name in ("TEMP", "TMP"))):
        raise ValueError("CPU/preallocation/UTF8/D-drive temporary environment differs from the registered guard")
    return dict(expected_commit=expected_commit, manifest_path=str(manifest_path),
                manifest_sha256=manifest_sha, source_files=len(manifest), guard_path=str(guard_path),
                guard_prefix_sha256=sha(content), hard_limit_seconds=60)


def load_runtime():
    global np, jax, jnp, seed, direct, ravel_pytree
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from seed import seed, direct
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise ValueError("probe requires the registered CPU float32 runtime; do not change flags to make it pass")
    actual = dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
                  jaxlib=importlib.metadata.version("jaxlib"))
    if actual != PINNED_RUNTIME:
        raise ValueError("runtime package versions differ from the pinned reference")


def shapes(c):
    _, inputs, _, neurons, _ = seed.dims(c)
    f32 = lambda shape: jax.ShapeDtypeStruct(shape, np.float32)
    abstract_genome = jax.eval_shape(lambda key: seed.init_genome(c, key),
                                    jax.ShapeDtypeStruct((2,), np.uint32))
    genomes = jax.tree.map(lambda value: f32((BATCH,) + value.shape), abstract_genome)
    common = (f32((BATCH, 1, neurons, neurons)), f32((BATCH, 1, neurons)),
              f32((BATCH, 1, inputs, c.H)), f32((BATCH, 1, inputs)))
    return dict(plastic=(genomes, *common, f32((BATCH, 1)), common[0]), direct=common)


def plastic_function(c, core):
    types = seed.neuron_types(c)

    def family(g, W, x, Wp, obs, denergy, mask):
        hebb = g["hebb"][types[:, None], types[None, :]]
        result = jax.vmap(lambda w, state, predictor, observation, reward, mk: core(
            w, state, predictor, observation, reward, mk, hebb, jnp.exp(g["log_eta"]),
            jnp.exp(g["log_plr"]), g["mod"], 1., c, predictor_plastic=1.))(
                W, x, Wp, obs, denergy, mask)
        return dict(zip(("W", "x", "Wp", "surprise"), result))

    return jax.jit(jax.vmap(family))


def direct_function(c, crop):
    _, inputs, _, _, _ = seed.dims(c)

    def family(W, x, Wp, obs):
        x = x.at[:, :inputs].set(obs)
        for _ in range(c.micro):
            if crop:
                hidden_motor = jax.vmap(lambda w, state: jnp.tanh(w[inputs:, :] @ state))(W, x)
                x = jnp.concatenate((obs, hidden_motor), axis=1)
            else:
                # Exact direct branch: no plastic clipping and no predictor update.
                x = jax.vmap(lambda w, state: jnp.tanh(w @ state))(W, x).at[:, :inputs].set(obs)
        return dict(W=W, x=x, Wp=Wp, surprise=jnp.mean(obs ** 2, axis=1))

    return jax.jit(jax.vmap(family))


def candidate_core():
    source = inspect.getsource(seed.brain_step)
    if source.count(ORIGINAL) != 1:
        raise ValueError("frozen brain_step no longer has the exact reviewed microstep; qualification must stop")
    replacement = source.replace(ORIGINAL, REPLACEMENT)
    namespace = dict(seed.__dict__)
    # A private function copy only. The imported production module is never patched.
    exec(compile(replacement, "<development sensor-row candidate>", "exec"), namespace)
    return namespace["brain_step"], source, replacement


def inspect_hlo(text, neurons=157, outputs=54, expected_dots=2):
    """Conservative shape evidence only; unknown/custom-call lowering never proves pruning."""
    if text is None:
        return dict(status="inconclusive", reason="compiled executable exposes no HLO text", neural_dots=[])
    instruction = re.compile(r"^\s*(?:ROOT\s+)?(%?[^\s=]+)\s*=\s*\w+\[([0-9,]*)\](?:\{[^}]*\})?\s+([\w-]+)\(")
    definitions, dots, unparsed = {}, [], []
    for line in text.splitlines():
        match = instruction.match(line)
        if match:
            name, dims, op = match.groups()
            definitions[name.lstrip("%")] = tuple(int(value) for value in dims.split(",") if value)
            if op == "dot":
                dots.append(line)
        elif re.search(r"\bdot\(", line):
            unparsed.append(line[:500])
    neural = []
    for line in dots:
        operands = re.search(r"\bdot\((.*?)\)", line)
        names = re.findall(r"%([\w.-]+)", operands.group(1)) if operands else []
        contract, batches = [], []
        for side in ("lhs", "rhs"):
            contract_match = re.search(side + r"_contracting_dims=\{([0-9,]*)\}", line)
            batch_match = re.search(side + r"_batch_dims=\{([0-9,]*)\}", line)
            contract.append([] if not contract_match else [int(x) for x in contract_match[1].split(",") if x])
            batches.append([] if not batch_match else [int(x) for x in batch_match[1].split(",") if x])
        if len(names) != 2 or any(name not in definitions for name in names) or any(len(value) != 1 for value in contract):
            unparsed.append(line[:500])
            continue
        left, right = (definitions[name] for name in names)
        if any(axis >= len(shape) for shape, axes in zip((left, right), contract) for axis in axes):
            unparsed.append(line[:500])
            continue
        if left[contract[0][0]] != neurons or right[contract[1][0]] != neurons:
            continue  # Predictor contraction is H48, not the recurrent N157 contraction.
        remaining = [size for shape, axes, batch in zip((left, right), contract, batches)
                     for axis, size in enumerate(shape) if axis not in axes + batch and size != 1]
        neural.append(dict(output_axes=remaining, lhs=list(left), rhs=list(right), evidence=line[:900]))
    custom = [line[:500] for line in text.splitlines() if "custom-call(" in line]
    pruned = len(neural) == expected_dots and all(item["output_axes"] == [outputs] for item in neural)
    status = "already_pruned" if pruned and not unparsed and not custom else "not_pruned" if any(
        item["output_axes"] == [neurons] for item in neural) else "inconclusive"
    return dict(status=status, neural_dots=neural, unparsed_dot_lines=unparsed,
                custom_call_lines=custom, explanation="Only unambiguous optimized N157-to-54 dot shapes permit an early exit.")


def compile_record(function, specification, folder, name):
    started = time.monotonic()
    executable = function.lower(*specification).compile()
    compilation_seconds = time.monotonic() - started
    text = executable.as_text()
    if text is not None:
        write_text(folder / (name + ".optimized-hlo.txt"), text)
    return executable, dict(compile_seconds=compilation_seconds,
        hlo_sha256=None if text is None else sha(text.encode()), inspection=inspect_hlo(text))


def fixtures(c):
    """Fresh deterministic synthetic states; no world simulation or production artifacts."""
    _, inputs, outputs, neurons, _ = seed.dims(c)
    keys = jax.random.split(jax.random.PRNGKey(FIXTURE_SEED), 12)
    flat, unravel = ravel_pytree(seed.init_genome(c, keys[0]))
    noise = jax.random.normal(keys[1], (BATCH // 2, flat.size), dtype=jnp.float32)
    genomes = jax.vmap(unravel)(flat + .1 * jnp.concatenate((noise, -noise)))
    birth_keys = jax.random.split(keys[2], BATCH)
    W, mask = jax.jit(jax.vmap(lambda g, key: jax.vmap(lambda own: seed.develop(g, own, c))(
        jax.random.split(key, c.M))))(genomes, birth_keys)
    # Retain actual signed-zero mask entries while producing a non-birth legal W.
    W = jnp.where(mask > 0, jnp.clip(W + .04 * jax.random.normal(keys[3], W.shape), -c.wmax, c.wmax), W)
    x = jax.random.uniform(keys[4], (BATCH, 1, neurons), minval=-.8, maxval=.8)
    Wp = .03 * jax.random.normal(keys[5], (BATCH, 1, inputs, c.H))
    item = jax.random.randint(keys[6], (BATCH, 1, 25), 0, 5)
    visual = jnp.concatenate(((item == 1).astype(jnp.float32), (item == 2).astype(jnp.float32),
                              jnp.zeros((BATCH, 1, 50), jnp.float32)), axis=-1)
    scalars = jax.random.uniform(keys[7], (BATCH, 1, 3), minval=-.8, maxval=.8).at[:, :, 2].set(1.)
    obs = jnp.concatenate((visual, scalars), axis=-1)
    # Previous motor efference and recurrent hidden state are both nontrivial.
    actions = jax.random.randint(keys[8], (BATCH, 1), 0, outputs - 1)
    reward = jnp.linspace(-.25, .25, BATCH, dtype=jnp.float32)[:, None]
    obs = obs.at[:, :, -2].set(4. * reward)
    x = x.at[:, :, :inputs].set(jnp.roll(obs, 1, axis=0))
    x = x.at[:, :, neurons - outputs:neurons - 1].set(jax.nn.one_hot(actions, outputs - 1))
    original_direct = direct.init_genome(c, keys[9])
    epsilon = jax.random.normal(keys[10], (BATCH // 2, original_direct.size))
    direct_population = original_direct + .1 * jnp.concatenate((epsilon, -epsilon))
    decode = jax.jit(jax.vmap(lambda value: direct.decode(value, c)))
    normal = decode(direct_population)[:, None]
    large = decode(direct_population.at[:, 0].set(7.25).at[:, 1].set(-6.5))[:, None]
    return {"plastic": (genomes, W, x, Wp, obs, reward, mask),
            "direct_normal": (normal, x, jnp.zeros_like(Wp), obs),
            "direct_large_finite": (large, x, jnp.zeros_like(Wp), obs)}


def tree_manifest(tree):
    result = []
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        value = np.ascontiguousarray(value)
        if not np.isfinite(value).all():
            raise ValueError("probe input/output contains a nonfinite value")
        result.append(dict(path=jax.tree_util.keystr(path), shape=list(value.shape), dtype=value.dtype.str,
                           sha256=sha(value.tobytes()), max_abs=float(np.abs(value).max(initial=0)),
                           negative_zeros=int(np.count_nonzero((value == 0) & np.signbit(value)))))
    return result


def byte_compare(reference, candidate):
    records, passed = {}, True
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate output fields differ")
    for name in reference:
        old, new = np.ascontiguousarray(reference[name]), np.ascontiguousarray(candidate[name])
        shape_equal = old.shape == new.shape and old.dtype == new.dtype
        old_bytes, new_bytes = old.view(np.uint8).reshape(-1), new.view(np.uint8).reshape(-1)
        unequal = np.flatnonzero(old_bytes != new_bytes) if shape_equal else None
        identical = shape_equal and len(unequal) == 0
        passed = passed and identical
        records[name] = dict(identical=identical, shape=list(old.shape), dtype=old.dtype.str,
                            reference_sha256=sha(old.tobytes()), candidate_sha256=sha(new.tobytes()),
                            unequal_bytes=None if unequal is None else int(len(unequal)),
                            first_unequal_byte=None if unequal is None or not len(unequal) else int(unequal[0]))
    return passed, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("inspect", "parity"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    folder = scratch_path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(stage=args.stage, scientific_result=False, deployment_authorized=False,
                  full_es_parity_checked=False, throughput_measured=False, fixture_seed=FIXTURE_SEED)
    exit_code = 0
    try:
        report["guard"] = verify_guard(args)
        load_runtime()
        c = seed.Cfg(M=1)
        if seed.dims(c)[1:4] != (103, 6, 157) or c.H != 48 or c.micro != 2:
            raise ValueError("production controller dimensions changed; qualification must stop")
        direct_source = inspect.getsource(seed.rollout)
        direct_block = '''W, Wp = s["W"], s["Wp"]
            x = s["x"].at[:, :D].set(obs)
            for _ in range(c.micro):
                x = jax.vmap(lambda w, state: jnp.tanh(w @ state))(W, x).at[:, :D].set(obs)
            surprise = jnp.mean(obs ** 2, axis=1)'''
        if direct_source.count(direct_block) != 1:
            raise ValueError("the production direct microtick changed; qualification must stop")
        report.update(cfg=c._asdict(), batch=BATCH,
            runtime=dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
                jaxlib=importlib.metadata.version("jaxlib"), backend=jax.default_backend(),
                enable_x64=bool(jax.config.jax_enable_x64), precision=str(jax.config.jax_default_matmul_precision),
                prng=str(jax.config.jax_default_prng_impl), partitionable=bool(jax.config.jax_threefry_partitionable)),
            compiler_environment={name: os.environ.get(name) for name in
                ("XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION", "OMP_NUM_THREADS")},
            source_sha256={str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in
                (Path(seed.__file__).resolve(), Path(direct.__file__).resolve(), Path(__file__).resolve())})
        spec = shapes(c)
        old, old_reports = {}, {}
        old["plastic"], old_reports["plastic"] = compile_record(plastic_function(c, seed.brain_step), spec["plastic"], folder, "reference-plastic")
        old["direct"], old_reports["direct"] = compile_record(direct_function(c, False), spec["direct"], folder, "reference-direct")
        report["reference_compilation"] = old_reports
        pruned = all(value["inspection"]["status"] == "already_pruned" for value in old_reports.values())
        if pruned:
            report["decision"] = "STOP_ALREADY_PRUNED_NO_NEW_ROW_WORK_TO_REMOVE"
        elif args.stage == "inspect":
            report["decision"] = "INSPECTION_ONLY_NO_NUMERICAL_QUALIFICATION"
        else:
            core, original, replacement = candidate_core()
            write_text(folder / "reference-brain-step.py.txt", original)
            write_text(folder / "candidate-brain-step.py.txt", replacement)
            report["brain_step_sources"] = dict(reference_sha256=sha(original.encode()), candidate_sha256=sha(replacement.encode()))
            data = fixtures(c)
            new, new_reports, comparisons = {}, {}, {}
            for case, inputs in data.items():
                controller = "plastic" if case == "plastic" else "direct"
                if controller not in new:
                    function = plastic_function(c, core) if controller == "plastic" else direct_function(c, True)
                    new[controller], new_reports[controller] = compile_record(function, spec[controller], folder, "candidate-" + controller)
                before = tree_manifest(inputs)
                reference = jax.tree.map(np.asarray, old[controller](*inputs))
                candidate = jax.tree.map(np.asarray, new[controller](*inputs))
                tree_manifest(reference)
                tree_manifest(candidate)
                passed, comparison = byte_compare(reference, candidate)
                if before != tree_manifest(inputs):
                    raise ValueError("probe mutated its common input state")
                comparisons[case] = dict(passed=passed, input_manifest=before, outputs=comparison)
                if not passed:
                    report["decision"] = "FAIL_BYTE_MISMATCH_STOP"
                    exit_code = 1
                    break
            else:
                geometry_verified = all(value["inspection"]["status"] == "already_pruned" for value in new_reports.values())
                report["candidate_compiled_row_geometry_verified"] = geometry_verified
                report["decision"] = ("PASS_MICROTICK_ONLY_NOT_DEPLOYMENT" if geometry_verified
                                      else "INCONCLUSIVE_CANDIDATE_COMPILED_GEOMETRY_STOP")
                if not geometry_verified:
                    exit_code = 1
            report.update(candidate_compilation=new_reports, comparisons=comparisons)
        verify_source_manifest(Path(report["guard"]["manifest_path"]), report["guard"]["manifest_sha256"])
        report["source_manifest_unchanged"] = True
    except Exception as error:
        report.update(decision="ERROR_NO_QUALIFICATION", error_type=type(error).__name__, error=str(error))
        exit_code = 1
    report["elapsed_seconds"] = time.monotonic() - started
    write_text(folder / "summary.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(decision=report["decision"], summary=str(folder / "summary.json"),
                          deployment_authorized=False), allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
