"""Regression cases discovered during the September 2026 notebook audit."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import nbformat as nbf
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import worker_train_indices, footer_fingerprint, decode_test
from src.sweep import (MU_PROTOCOL, candidate_name, choose_winner, plan_sweep,
                       read_claims, selected_mu, selection_signature)
from src import ckpt as C
from tests.run_e2e import make_cfg

class ReviewRegression(unittest.TestCase):
    def test_calibration_projection_uses_the_configured_eval_path(self):
        from scripts.gen_notebooks import CALIB_BENCH
        code = CALIB_BENCH.split('# ---- 4. project the three scenarios')[1]
        code = code.split('del Xg,')[0].split('\n', 1)[1]
        report = {'eval': {'folded': {16384: {'client_rows_per_s': 1000.}},
                           'vmap': {8: {'client_rows_per_s': 1000000.,
                                       'cm_proto_mismatch': 100}}},
                  'step_ms': {'512|reduce-overhead|amp=True': 'FAILED',
                              '512|eager|amp=True': 10., '256|eager|amp=True': 20.}}
        g = {'report': report, 'REF_BS': 16384, 'CFG': {'compile': True},
             'print': lambda *a, **kw: None}
        exec(compile(code, '<calibration projection>', 'exec'), g)
        self.assertEqual(set(report['projection']), {20, 50, 100})
        self.assertEqual(report['projection'][20]['train_s'], 84083 * .01 / 2)
        self.assertEqual(report['projection'][100]['eval_s'], 100 * 10761343 / 1000 / 2)

    def test_holdout_indices_stay_in_their_client(self):
        spans = {3: (0, 4), 8: (4, 10), 11: (10, 15)}
        local = {3: np.array([0, 2]), 8: np.array([1, 5]), 11: np.array([0, 4])}
        actual = worker_train_indices(local, spans)
        np.testing.assert_array_equal(actual[8], [5, 9])
        np.testing.assert_array_equal(actual[11], [10, 14])
        for cid, idx in actual.items():
            lo, hi = spans[cid]
            self.assertTrue(((idx >= lo) & (idx < hi)).all())
        with self.assertRaises(ValueError):
            worker_train_indices({8: np.array([6])}, {8: (4, 10)})

    def test_incomplete_grid_never_selects_a_winner(self):
        results = {'0.1': {'rounds': 4, 'last_proto_f1_macro': .7, 'mu_absolute': 1e-6},
                   '1.0': {'rounds': 2, 'last_proto_f1_macro': .9, 'mu_absolute': 1e-5}}
        self.assertIsNone(choose_winner(results, [.1, 1.], 4))
        results['1.0']['rounds'] = 4
        self.assertEqual(choose_winner(results, [.1, 1.], 4)['k'], 1.)
        results['1.0']['last_proto_f1_macro'] = float('nan')
        self.assertIsNone(choose_winner(results, [.1, 1.], 4))

    def test_rng_checkpoint_loads_with_weights_only(self):
        saved = C.rng_state()
        expected = (np.random.random(3), torch.rand(3))
        with tempfile.TemporaryDirectory() as td:
            f = Path(td)/'rng.pt'; C.atomic_save(saved, f)
            C.set_rng_state(torch.load(f, weights_only=True))
            np.testing.assert_array_equal(np.random.random(3), expected[0])
            self.assertTrue(torch.equal(torch.rand(3), expected[1]))

    def test_scientific_configuration_cannot_silently_resume(self):
        cfg, paths = make_cfg(Path('/tmp/tinyproto_fixture'), 4, 3, ['cuda:0', 'cuda:0'])
        cfg.update(scaler=json.loads(Path(paths['scaler']).read_text()), artifact_version=2,
                   validation_fingerprint='full-train-fixed-test')
        baseline = C.fingerprint(cfg)
        for field, value in [('lr', .05), ('mu_value', .3), ('lam', 4.),
                             ('model_cfg', {}), ('feature_cols', list(reversed(cfg['feature_cols']))),
                             ('class_names', list(reversed(cfg['class_names']))), ('scaler', {}),
                             ('validation_fingerprint', 'different-split'), ('assignment', [[0,1],[2,3]])]:
            c = copy.deepcopy(cfg); c[field] = value
            self.assertNotEqual(C.fingerprint(c), baseline, field)
        c = dict(cfg, max_hours=1, require_resume=True)
        self.assertEqual(C.fingerprint(c), baseline)
        doc = _valid_sweep_doc(cfg)
        self.assertEqual(selected_mu(doc, cfg)[0], doc['winner']['mu_absolute'])
        with self.assertRaises(ValueError):
            selected_mu(doc, dict(cfg, scenario='100client'))

    def test_footer_identity_is_stable_and_mount_independent(self):
        import shutil
        files = sorted(Path('/tmp/tinyproto_fixture/train').rglob('*.parquet'))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'train'; shutil.copytree('/tmp/tinyproto_fixture/train', root)
            self.assertEqual(footer_fingerprint(files), footer_fingerprint(sorted(root.rglob('*.parquet'))))


def _valid_sweep_doc(cfg, **override):
    """A mu_sweep.json that satisfies the frozen protocol, so tests can break one thing at a time."""
    grid = MU_PROTOCOL["grid"]
    results = {str(k): {"rounds": MU_PROTOCOL["rounds"], "mu_absolute": 1e-5 * (i + 1),
                        "last_proto_f1_macro": 0.10 * (i + 1),
                        "best_proto_f1_macro": 0.10 * (i + 1)}
               for i, k in enumerate(grid)}
    best = grid[-1]
    doc = {"scenario": cfg["scenario"], "grid": list(grid), "rounds": MU_PROTOCOL["rounds"],
           "protocol": dict(MU_PROTOCOL), "selection_signature": selection_signature(cfg),
           "results": results,
           "validation_fingerprint": {"fraction": MU_PROTOCOL["val_fraction"],
                                      "seed": MU_PROTOCOL["val_seed"]},
           "status": "complete",
           "winner": {"k": float(best), "mu_absolute": results[str(best)]["mu_absolute"],
                      "criterion": MU_PROTOCOL["criterion"]}}
    doc.update(override)
    return doc


class MuProtocolContract(unittest.TestCase):
    """R1: production must verify the sweep ran THIS procedure, not merely that one finished."""

    def setUp(self):
        self.cfg, _ = make_cfg(Path('/tmp/tinyproto_fixture'), 4, 4, ['cpu'])
        self.cfg.setdefault('scaler_fingerprint', 'deadbeef')
        self.cfg.setdefault('data_fingerprint', 'cafe')

    def test_valid_document_is_accepted(self):
        doc = _valid_sweep_doc(self.cfg)
        self.assertEqual(selected_mu(doc, self.cfg)[0], doc['winner']['mu_absolute'])

    def test_rejects_documents_that_did_not_run_the_protocol(self):
        cfg = self.cfg
        cases = {
            'shortened sweep': _valid_sweep_doc(cfg, rounds=1),
            'single-candidate grid': _valid_sweep_doc(cfg, grid=[1.0]),
            'no results': _valid_sweep_doc(cfg, results={}),
            'missing one candidate': _valid_sweep_doc(
                cfg, results={k: v for k, v in _valid_sweep_doc(cfg)['results'].items()
                              if k != str(MU_PROTOCOL['grid'][0])}),
            'different validation fraction': _valid_sweep_doc(
                cfg, validation_fingerprint={'fraction': 0.5, 'seed': 42}),
            'different validation seed': _valid_sweep_doc(
                cfg, validation_fingerprint={'fraction': 0.02, 'seed': 7}),
            'no protocol block': _valid_sweep_doc(cfg, protocol=None),
            'older protocol version': _valid_sweep_doc(
                cfg, protocol=dict(MU_PROTOCOL, version=0)),
            'different criterion': _valid_sweep_doc(
                cfg, protocol=dict(MU_PROTOCOL, criterion='best round')),
        }
        for name, doc in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    selected_mu(doc, cfg)

    def test_rejects_a_winner_that_the_results_do_not_support(self):
        grid = MU_PROTOCOL['grid']
        # winner not in the grid at all
        doc = _valid_sweep_doc(self.cfg)
        doc['winner'] = dict(doc['winner'], k=99.0)
        with self.assertRaises(ValueError):
            selected_mu(doc, self.cfg)
        # winner names a real candidate, but not the best-scoring one
        doc = _valid_sweep_doc(self.cfg)
        loser = str(grid[0])
        doc['winner'] = {'k': float(grid[0]),
                         'mu_absolute': doc['results'][loser]['mu_absolute'],
                         'criterion': MU_PROTOCOL['criterion']}
        with self.assertRaises(ValueError):
            selected_mu(doc, self.cfg)
        # winner k is right but the mu value was edited
        doc = _valid_sweep_doc(self.cfg)
        doc['winner'] = dict(doc['winner'], mu_absolute=doc['winner']['mu_absolute'] * 2)
        with self.assertRaises(ValueError):
            selected_mu(doc, self.cfg)

    def test_signature_covers_the_scaler(self):
        a = selection_signature(self.cfg)
        b = selection_signature(dict(self.cfg, scaler_fingerprint='0' * 8))
        self.assertNotEqual(a, b)

    def test_signature_ignores_round_count_so_sweep_and_production_match(self):
        self.assertEqual(selection_signature(dict(self.cfg, rounds=4)),
                         selection_signature(dict(self.cfg, rounds=50)))

    def test_knob_cell_matches_the_frozen_protocol(self):
        from scripts.gen_notebooks import MU_CFG
        g = {}
        exec(compile(MU_CFG.split('CFG = {')[0], '<mu knobs>', 'exec'), g)
        self.assertEqual(g['MU_GRID'], MU_PROTOCOL['grid'])
        self.assertEqual(g['SWEEP_ROUNDS'], MU_PROTOCOL['rounds'])
        self.assertEqual(g['VAL_FRAC'], MU_PROTOCOL['val_fraction'])
        self.assertEqual(g['VAL_SEED'], MU_PROTOCOL['val_seed'])


class SweepResumeIntegrity(unittest.TestCase):
    """R2: a partial attachment must stop the sweep, not restart the missing candidate."""

    def _commit(self, root, scenario, k, rounds):
        d = Path(root) / candidate_name(scenario, k) / 'complete'
        d.mkdir(parents=True, exist_ok=True)
        for r in range(1, rounds + 1):
            (d / f'round_{r:03d}.done').write_text('')

    def test_never_started_candidates_are_not_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            plan = plan_sweep(MU_PROTOCOL['grid'], '20client', [Path(td)])
            self.assertEqual({v['committed_rounds'] for v in plan.values()}, {0})

    def test_partial_attachment_is_refused(self):
        grid = MU_PROTOCOL['grid']
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'runs'
            self._commit(root, '20client', grid[1], 4)          # B survived
            claims = {str(grid[0]): 4, str(grid[1]): 4}         # record says A finished too
            with self.assertRaises(RuntimeError) as e:
                plan_sweep(grid, '20client', [root], claims=claims)
            self.assertIn(str(grid[0]), str(e.exception))

    def test_claims_are_read_from_a_previous_summary(self):
        grid = MU_PROTOCOL['grid']
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / 'prev'
            base.mkdir(parents=True)
            (base / 'mu_sweep_progress.json').write_text(json.dumps({
                'scenario': '20client',
                'candidates': {str(grid[0]): {'committed_rounds': 4}}}))
            self.assertEqual(read_claims('20client', [base]), {str(grid[0]): 4})
            # a different scenario's record must not be picked up
            self.assertEqual(read_claims('50client', [base]), {})

    def test_scenarios_do_not_cross_match(self):
        grid = MU_PROTOCOL['grid']
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'runs'
            self._commit(root, '20client', grid[0], 4)
            plan = plan_sweep(grid, '50client', [root])
            self.assertEqual(plan[str(grid[0])]['committed_rounds'], 0)


class SourceManifestContract(unittest.TestCase):
    """P6: the config fingerprint covers the science but not the code. Editing src/ between two
    sessions of one run keeps the fingerprint identical and passes the validator."""

    ROOT = Path(__file__).resolve().parents[1]

    def _run(self, *args, cwd=None):
        import subprocess
        return subprocess.run([sys.executable, str(self.ROOT / "scripts/source_manifest.py"),
                               *args], capture_output=True, text=True)

    def test_verify_passes_on_the_pinned_tree(self):
        r = self._run("verify")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_verify_detects_a_changed_module(self):
        with tempfile.TemporaryDirectory() as td:
            man = Path(td) / "m.json"
            self.assertEqual(self._run("write", "--manifest", str(man)).returncode, 0)
            doc = json.loads(man.read_text())
            doc["files"]["src/driver.py"] = "0" * 64      # as if driver.py had been edited
            man.write_text(json.dumps(doc))
            r = self._run("verify", "--manifest", str(man))
            self.assertNotEqual(r.returncode, 0, "a changed module must fail verification")
            self.assertIn("src/driver.py", r.stdout)

    def test_verify_detects_a_changed_runtime_image(self):
        with tempfile.TemporaryDirectory() as td:
            man = Path(td) / "m.json"
            self._run("write", "--manifest", str(man))
            doc = json.loads(man.read_text())
            doc["docker_image"] = "gcr.io/kaggle-images/python@sha256:" + "0" * 64
            man.write_text(json.dumps(doc))
            r = self._run("verify", "--manifest", str(man))
            self.assertNotEqual(r.returncode, 0, "a changed runtime image must fail verification")
            self.assertIn("docker_image", r.stdout)


class SessionChainContract(unittest.TestCase):
    """P3: four broken session notebooks that the validator used to accept with '0 problems'.
    A -s2 that resumes nothing does not fail loudly -- it quietly trains from round 1 again."""

    ROOT = Path(__file__).resolve().parents[1]
    PROD = ROOT / "papers/tinyproto-lee-2026/production"

    def _validate(self, d):
        import subprocess
        return subprocess.run([sys.executable, str(self.ROOT / "scripts/validate_notebooks.py"),
                               str(d)], capture_output=True, text=True)

    def test_real_production_set_passes(self):
        if not self.PROD.exists():
            self.skipTest("production set not generated")
        r = self._validate(self.PROD)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_broken_session_chains_are_rejected(self):
        import shutil
        src = self.PROD / "tinyproto-fp-train-100client-s2"
        if not src.exists():
            self.skipTest("production set not generated")
        n = "100client"
        # s2 now resumes by DATASET (its owner cannot read s1's kernel output), so a mutation
        # that only touches kernel_sources leaves the dataset route intact and the notebook
        # still valid. Strip the resume dataset first, then mutate, so each case tests what its
        # name says.
        owner = json.loads((src / "kernel-metadata.json").read_text())["id"].split("/")[0]

        def drop_resume_ds(m):
            m["dataset_sources"] = [d for d in m["dataset_sources"]
                                    if "tinyproto-fp-resume-" not in d]

        def ks(*v):
            def f(m):
                drop_resume_ds(m)
                m["kernel_sources"] = list(v)
            return f

        mutations = {
            # 1. session slug says s2 but nothing is resumed -> would train from round 1
            "s2 resumes nothing": ks(),
            # 2. a notebook attaching its own output
            "s2 attaches itself": ks(f"{owner}/tinyproto-fp-train-{n}-s2"),
            # 3. resume source from a different scenario
            "resumes another scenario": ks(f"{owner}/tinyproto-fp-train-50client-s1"),
            # 4. a third dataset that carries no mu at all
            "third dataset is not mu": lambda m: m.__setitem__(
                "dataset_sources", [d for d in m["dataset_sources"] if "-mu-" not in d]
                + ["odixe0502/some-unrelated-dataset"]),
            # 5. mu for the wrong scenario
            "mu for another scenario": lambda m: m.__setitem__(
                "dataset_sources", [d for d in m["dataset_sources"] if "-mu-" not in d]
                + ["odixe0502/tinyproto-fp-mu-20client"]),
            # 6. skipping a session in the chain
            "s2 continues s0": ks(f"{owner}/tinyproto-fp-train-{n}-s0"),
            # 7. a CROSS-ACCOUNT kernel source. Measured 2026-09-09: Kaggle drops it at push
            #    time with one warning line, still reports the push as successful, and mounts an
            #    empty /kaggle/input. This exact layout was the plan for 100client s3 -> s4 and
            #    would have been discovered ~30 h of quota later.
            "resumes another ACCOUNT's kernel output":
                ks(f"khanhmay0304/tinyproto-fp-train-{n}-s1"),
            # 8. both routes at once -> two candidates with one fingerprint, which
            #    find_import_source refuses to choose between.
            "both resume routes": lambda m: m.__setitem__(
                "kernel_sources", [f"{owner}/tinyproto-fp-train-{n}-s1"]),
            # 9. a resume dataset belonging to a different scenario
            "resume dataset of another scenario": lambda m: (
                drop_resume_ds(m),
                m["dataset_sources"].append(f"{owner}/tinyproto-fp-resume-50client-r31")),
        }
        with tempfile.TemporaryDirectory() as td:
            for name, mut in mutations.items():
                with self.subTest(name):
                    d = Path(td) / name.replace(" ", "_") / src.name
                    shutil.copytree(src, d)
                    m = json.loads((d / "kernel-metadata.json").read_text())
                    mut(m)
                    (d / "kernel-metadata.json").write_text(json.dumps(m, indent=2))
                    r = self._validate(d.parent)
                    self.assertNotEqual(r.returncode, 0, f"validator accepted: {name}")

    def test_dataset_resume_route_is_accepted_for_a_cross_account_session(self):
        """The only route that works across accounts. It must satisfy the same session-chain
        promise as a kernel source: require_resume set, no kernel source, exactly one dataset."""
        src = self.PROD / "tinyproto-fp-train-100client-s2"
        if not src.exists():
            self.skipTest("production set not generated")
        m = json.loads((src / "kernel-metadata.json").read_text())
        self.assertEqual(m["kernel_sources"], [], "s2 must not attach a kernel source")
        ds = [d for d in m["dataset_sources"] if "tinyproto-fp-resume-" in d]
        self.assertEqual(len(ds), 1, f"expected exactly one resume dataset, got {ds}")
        self.assertTrue(ds[0].startswith(m["id"].split("/")[0] + "/"),
                        f"the resume dataset {ds[0]} is not owned by the account that runs it, "
                        "so it is not readable either")
        body = "\n".join(c.source for c in
                         nbf.read(str(src / f"{src.name}.ipynb"), as_version=4).cells)
        self.assertIn('"require_resume": True', body)
        self.assertEqual(self._validate(self.PROD).returncode, 0)

    def test_session_one_must_not_attach_a_previous_session(self):
        import shutil
        src = self.PROD / "tinyproto-fp-train-100client-s1"
        if not src.exists():
            self.skipTest("production set not generated")
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "s1_with_prev" / src.name
            shutil.copytree(src, d)
            m = json.loads((d / "kernel-metadata.json").read_text())
            m["kernel_sources"] = ["khanhmay0304/tinyproto-fp-train-100client-s3"]
            (d / "kernel-metadata.json").write_text(json.dumps(m, indent=2))
            self.assertNotEqual(self._validate(d.parent).returncode, 0,
                                "a session-1 notebook must not attach a previous session")


class TelemetryNeverStopsTraining(unittest.TestCase):
    """P1: wandb_run.log sits between the round commit and the timing write. An exception there
    used to escape, killing a run whose round was already safely on disk and losing the timing
    the NEXT session's budget gate reads back."""

    def test_failing_logger_does_not_stop_the_run(self):
        fixture = Path("/tmp/tinyproto_fixture")
        if not fixture.exists():
            self.skipTest("fixture missing; build with tests/make_fixture.py")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tests.run_e2e import make_cfg
        from src.driver import run

        class ExplodingLogger:
            def __init__(self): self.calls = 0
            def log(self, *a, **k):
                self.calls += 1
                raise RuntimeError("simulated W&B outage")

        wb = ExplodingLogger()
        with tempfile.TemporaryDirectory() as td:
            cfg, paths = make_cfg(fixture, 4, 2, ["cpu"], run_name="p1_telemetry",
                                  extra={"amp": False, "compile": False})
            res = run(cfg, paths, Path(td), log=lambda *a, **k: None, wandb_run=wb)
            d = Path(td) / "runs" / "p1_telemetry"
            self.assertEqual(res["last_round"], 2, "training must reach the final round")
            for rnd in (1, 2):
                self.assertTrue((d / "complete" / f"round_{rnd:03d}.done").exists(),
                                f"round {rnd} was not committed")
                self.assertTrue((d / "logs" / f"timing_{rnd:03d}.json").exists(),
                                f"timing for round {rnd} was lost to the telemetry failure")
            self.assertEqual(wb.calls, 1,
                             "telemetry must be disabled after the first failure, not retried")


class AmpProbeGate(unittest.TestCase):
    """P5: the decisive-margin threshold scales with the measured delta, so a large delta used
    to widen the margin until no row qualified and the argmax test passed on an empty set."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import torch
        from src.fl_worker import probe_verdict
        self.torch, self.verdict = torch, probe_verdict

    def test_rejects_the_vacuous_pass(self):
        """The exact counterexample from the review: delta=1000, decisive=0, disagreements=0."""
        t = self.torch
        ref = t.tensor([[2.0, 0.0]])
        got = t.tensor([[0.0, 1000.0]])
        with self.assertRaises(RuntimeError) as cm:
            self.verdict(ref, got, min_decisive_frac=0.5, max_delta=0.05, label="autocast")
        self.assertIn("ceiling", str(cm.exception))

    def test_rejects_insufficient_coverage(self):
        """Delta small but almost no row has a decisive margin: proves nothing either way."""
        t = self.torch
        ref = t.zeros(100, 2)
        ref[0, 0] = 5.0                       # exactly one decisive row out of a hundred
        got = ref.clone()
        with self.assertRaises(RuntimeError) as cm:
            self.verdict(ref, got, min_decisive_frac=0.5, label="compile")
        self.assertIn("inconclusive", str(cm.exception))

    def test_rejects_a_real_argmax_disagreement(self):
        t = self.torch
        # Well-separated rows (margin 0.4, far above the fixed 0.01 threshold) whose argmax
        # moves. Under the old margin>10*delta rule this set was empty and the check passed.
        ref = t.tensor([[0.2, -0.2]] * 10)
        got = t.tensor([[-0.2, 0.2]] * 10)
        with self.assertRaises(RuntimeError) as cm:
            self.verdict(ref, got, min_decisive_frac=0.5, label="compile")
        self.assertIn("decisive rows", str(cm.exception))

    def test_accepts_a_realistic_fp16_probe(self):
        """Measured T4 behaviour: delta ~5e-4, ~95% of rows decisive, argmax stable."""
        t = self.torch
        t.manual_seed(0)
        ref = t.randn(256, 16) * 3
        got = ref + t.randn(256, 16) * 1e-4
        out = self.verdict(ref, got, min_decisive_frac=0.5, max_delta=0.05, label="autocast")
        self.assertEqual(out["disagreements"], 0)
        self.assertGreaterEqual(out["decisive_frac"], 0.5)


class CrossAccountResumeContract(unittest.TestCase):
    """A production session may run on an account that cannot read the sweep owner's private
    output. mu then arrives via a shared dataset.

    The previous session CANNOT arrive by kernel_sources across accounts. Probed on Kaggle
    2026-09-09 (minhtrit06 attaching khanhmay0304's private notebook output): the push prints
    'not valid kernel sources', reports success anyway, and the kernel starts with an EMPTY
    /kaggle/input. This class asserted the opposite until that probe ran."""

    ROOT = Path(__file__).resolve().parents[1]
    MU_DS = 'odixe0502/tinyproto-fp-mu-100client'

    def _gen(self, out, *extra):
        import subprocess
        r = subprocess.run([sys.executable, str(self.ROOT / 'scripts/gen_notebooks.py'),
                            '--out', str(out), *extra], capture_output=True, text=True)
        return r.returncode, r.stderr

    def _validate(self, d):
        import subprocess
        return subprocess.run([sys.executable, str(self.ROOT / 'scripts/validate_notebooks.py'),
                               str(d)], capture_output=True, text=True).returncode

    RESUME_DS = 'minhtrit06/tinyproto-fp-resume-100client-r16'

    def test_cross_account_session_resumes_by_dataset(self):
        """The route that actually works: the previous session's tree as a dataset owned by the
        account that will run it."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'nb'
            rc, err = self._gen(out, '--owner', 'minhtrit06', '--mu-dataset', self.MU_DS,
                                '--only', 'train100', '--session', '2',
                                '--resume-dataset', self.RESUME_DS)
            self.assertEqual(rc, 0, err)
            self.assertEqual(self._validate(out), 0, 'dataset-resume notebook must validate')
            m = json.loads((out / 'tinyproto-fp-train-100client-s2' /
                            'kernel-metadata.json').read_text())
            self.assertEqual(m['id'], 'minhtrit06/tinyproto-fp-train-100client-s2')
            self.assertEqual(m['kernel_sources'], [],
                             'a dataset-resume session must attach no kernel source')
            self.assertIn(self.RESUME_DS, m['dataset_sources'])
            self.assertIn(self.MU_DS, m['dataset_sources'])
            body = "\n".join(c.source for c in nbf.read(
                str(out / 'tinyproto-fp-train-100client-s2' /
                    'tinyproto-fp-train-100client-s2.ipynb'), as_version=4).cells)
            self.assertIn('"require_resume": True', body)

    def test_cross_account_kernel_source_is_rejected(self):
        """The layout this class used to certify. It generates -- the generator cannot know who
        owns what -- but it must NOT validate, because Kaggle mounts nothing for it."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'nb'
            rc, err = self._gen(out, '--owner', 'minhtrit06', '--mu-dataset', self.MU_DS,
                                '--only', 'train100', '--session', '4', '--resume-from',
                                'khanhmay0304/tinyproto-fp-train-100client-s3')
            self.assertEqual(rc, 0, err)
            self.assertNotEqual(self._validate(out), 0,
                                'a cross-account kernel source must be rejected: Kaggle drops '
                                'it at push time and the session starts from round 1')

    def test_the_two_resume_routes_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as td:
            rc, err = self._gen(Path(td) / 'nb', '--owner', 'minhtrit06', '--only', 'train100',
                                '--mu-dataset', self.MU_DS, '--session', '2',
                                '--resume-from', 'minhtrit06/tinyproto-fp-train-100client-s1',
                                '--resume-dataset', self.RESUME_DS)
            self.assertNotEqual(rc, 0)
            self.assertIn('pass one', err)

    def test_resume_from_refuses_to_span_scenarios(self):
        """One resume source belongs to one scenario; attaching it to all three would let
        find_import_source import the wrong run."""
        with tempfile.TemporaryDirectory() as td:
            rc, err = self._gen(Path(td) / 'nb', '--resume-from', 'odixe0502/x')
            self.assertNotEqual(rc, 0)
            self.assertIn('--only', err)

    def test_validator_rejects_broken_cross_account_metadata(self):
        import shutil
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'nb'
            rc, err = self._gen(out, '--owner', 'minhtrit06', '--mu-dataset', self.MU_DS,
                                '--only', 'train100', '--resume-from',
                                'khanhmay0304/tinyproto-fp-train-100client')
            self.assertEqual(rc, 0, err)
            src = out / 'tinyproto-fp-train-100client'
            mutations = {
                'two resume sources': lambda m: m.__setitem__('kernel_sources', [
                    'khanhmay0304/tinyproto-fp-train-100client',
                    'odixe0502/tinyproto-fp-train-100client']),
                'resume dropped but require_resume still set': lambda m: m.__setitem__(
                    'kernel_sources', []),
                'foreign sweep attached': lambda m: m.__setitem__('kernel_sources', [
                    'khanhmay0304/tinyproto-fp-train-100client',
                    'odixe0502/tinyproto-fp-mu-sweep-50client']),
                'mu unreachable: dataset removed': lambda m: m.__setitem__(
                    'dataset_sources', [d for d in m['dataset_sources'] if d != self.MU_DS]),
            }
            for name, mut in mutations.items():
                with self.subTest(name):
                    d = Path(td) / name.replace(' ', '_') / src.name
                    shutil.copytree(src, d)
                    m = json.loads((d / 'kernel-metadata.json').read_text())
                    mut(m)
                    (d / 'kernel-metadata.json').write_text(json.dumps(m, indent=2))
                    self.assertNotEqual(self._validate(d.parent), 0,
                                        f'validator accepted: {name}')


class NotebookRuntimeContract(unittest.TestCase):
    """R3: the validator must fail when a runtime guarantee is removed."""

    NB = Path(__file__).resolve().parents[1] / 'papers/tinyproto-lee-2026/notebook/tinyproto-fp-train-20client'

    def _validate(self, d):
        import subprocess
        r = subprocess.run([sys.executable,
                            str(Path(__file__).resolve().parents[1] / 'scripts/validate_notebooks.py'),
                            str(d)], capture_output=True, text=True)
        return r.returncode

    def test_each_runtime_guarantee_is_enforced(self):
        import shutil
        import nbformat as nbf
        mutations = {
            'drop docker_image': lambda m, nb: m.pop('docker_image', None),
            'drop notebook accelerator': lambda m, nb: nb.metadata.pop('kaggle', None),
            'gpu disabled in notebook': lambda m, nb: nb.metadata['kaggle'].__setitem__('isGpuEnabled', False),
            'wrong python version': lambda m, nb: nb.metadata.__setitem__(
                'language_info', {'name': 'python', 'version': '3.11'}),
            'sweep not attached': lambda m, nb: m.__setitem__('kernel_sources', []),
            'wrong sweep attached': lambda m, nb: m.__setitem__(
                'kernel_sources', ['odixe0502/tinyproto-fp-mu-sweep-50client']),
        }
        with tempfile.TemporaryDirectory() as td:
            unmodified = Path(td) / 'ok' / self.NB.name
            shutil.copytree(self.NB, unmodified)
            self.assertEqual(self._validate(unmodified.parent), 0, 'unmodified notebook must pass')
            for name, mut in mutations.items():
                with self.subTest(name):
                    d = Path(td) / name.replace(' ', '_') / self.NB.name
                    shutil.copytree(self.NB, d)
                    m = json.loads((d / 'kernel-metadata.json').read_text())
                    nb = nbf.read(d / m['code_file'], as_version=4)
                    mut(m, nb)
                    (d / 'kernel-metadata.json').write_text(json.dumps(m, indent=2))
                    nbf.write(nb, d / m['code_file'])
                    self.assertNotEqual(self._validate(d.parent), 0, f'validator accepted: {name}')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
