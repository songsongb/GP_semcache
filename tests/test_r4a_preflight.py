"""Branch gates only; mocked allocation/dependencies never execute a GPU model."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import ProtocolError


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/102_run_r4a_opt_full_e2e.py'
spec = importlib.util.spec_from_file_location('r4a_preflight_test_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BranchPreflight(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            'SLURM_JOB_ID': 'TEST_FIXTURE_ONLY_BRANCH_GATE',
            'CUDA_VISIBLE_DEVICES': '0',
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_r4_branch_accepts_with_available_dependencies(self):
        package = ModuleType('sacrebleu')
        package.__path__ = []
        metrics = ModuleType('sacrebleu.metrics')
        package.metrics = metrics
        with patch.object(runner, 'source_identity', return_value={'git_branch': 'exp/raw-semcache-r4'}), \
                patch.object(runner.importlib.util, 'find_spec', return_value=object()) as dependencies, \
                patch.dict(sys.modules, {'sacrebleu': package, 'sacrebleu.metrics': metrics}):
            runner.gpu_preflight()
        self.assertEqual([call.args[0] for call in dependencies.call_args_list],
                         ['torch', 'numpy', 'transformers', 'peft', 'sacrebleu'])

    def test_previous_r3_branch_rejects_before_dependency_checks(self):
        self.assert_branch_rejected('exp/raw-semcache-r3')

    def test_unrelated_or_detached_branch_rejects_before_dependency_checks(self):
        for branch in ('main', 'exp/raw-semcache-r4-other', '', None):
            with self.subTest(branch=branch):
                self.assert_branch_rejected(branch)

    def assert_branch_rejected(self, branch):
        with patch.object(runner, 'source_identity', return_value={'git_branch': branch}), \
                patch.object(runner.importlib.util, 'find_spec') as dependencies:
            with self.assertRaisesRegex(ProtocolError, r'wrong branch: R4-A requires exp/raw-semcache-r4'):
                runner.gpu_preflight()
            dependencies.assert_not_called()


if __name__ == '__main__':
    unittest.main()
