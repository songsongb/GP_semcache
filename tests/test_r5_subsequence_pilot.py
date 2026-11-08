"""Actual job argv/parser and immutable CPU approval gates; no GPU launched."""
import contextlib,importlib.util,io,os,shlex,subprocess,unittest
from pathlib import Path
from unittest.mock import patch

class PilotCLI(unittest.TestCase):
    def cli(self):
        spec=importlib.util.spec_from_file_location('subsequence_cli',Path(__file__).resolve().parents[1]/'scripts/151_r5_subsequence_generation.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli);return cli
    def test_actual_slurm_command_matches_public_parser(self):
        job=(Path(__file__).resolve().parents[1]/'jobs/r5_subsequence_generation.sh').read_text()
        argv=shlex.split(job.split('python -u scripts/151_r5_subsequence_generation.py ',1)[1].replace('\\\n',' '))
        cli=self.cli()
        with patch('semcache.edgelora.r5_subsequence_pilot.run',return_value={'TEST_FIXTURE_ONLY':True}) as run,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(argv),0);self.assertEqual(run.call_args.kwargs['cpu_audit'],'$SSG_CPU_AUDIT');self.assertTrue(run.call_args.kwargs['enable_gpu_experiment'])
        pos=argv.index('--cpu-audit');bad=argv[:pos]+argv[pos+2:]
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):cli.main(bad)
    def test_real_shell_rejects_before_dispatch_without_allocation(self):
        job=Path(__file__).resolve().parents[1]/'jobs/r5_subsequence_generation.sh'
        r=subprocess.run(['bash',str(job)],env={'PATH':os.environ['PATH']},capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('manual allocation required',r.stderr)
    def test_prepare_arguments_dispatch_without_gpu(self):
        with patch('semcache.edgelora.r5_subsequence_pilot.prepare',return_value={'TEST_FIXTURE_ONLY':True}) as prepare,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.cli().main(['prepare','--registry','r','--multi','m','--source-commit','s','--output-dir','o','--approximate']),0)
            self.assertEqual(prepare.call_args.kwargs,dict(registry='r',multi='m',source_commit='s',output_dir='o',approximate=True))
    def test_interrupted_worker_preserves_primary_and_terminates(self):
        from semcache.edgelora.r5_subsequence_pilot import wait_worker
        from unittest.mock import Mock
        child=Mock();primary=TimeoutError('original deadline');child.wait.side_effect=[primary,0]
        with patch('semcache.edgelora.r5_subsequence_pilot.subprocess.Popen',return_value=child):
            with self.assertRaises(TimeoutError) as raised:wait_worker(['CPU_FIXTURE_ONLY'],timeout=1)
        self.assertIs(raised.exception,primary);child.terminate.assert_called_once();self.assertEqual(child.wait.call_count,2)

if __name__=='__main__':unittest.main()
