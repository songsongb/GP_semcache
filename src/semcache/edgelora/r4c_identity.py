"""Immutable implementation identity shared by C0 preparation and C1 workers."""
from .communication import digest_bytes
from .r3a_artifacts import ROOT


def sources():
    paths=[*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'src').rglob('*.json')),
           ROOT/'scripts/105_prepare_r4c_longbench.py',ROOT/'scripts/106_run_r4c_quality_e2e.py',
           ROOT/'jobs/r4c_opt_quality_e2e.sh',ROOT/'docs/R4C_LONGBENCH_QUALITY_E2E.md']
    return {str(p.relative_to(ROOT)):digest_bytes(p.read_bytes()) for p in paths if p.is_file()}
