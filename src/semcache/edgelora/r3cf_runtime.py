"""Focused schedule injected only in an isolated NEW worker; original math/I/O."""
import time

from .r3a_artifacts import append,write
from .r3c_network_proxy import control
from .r3c_runtime import NetworkExperiment
from .r3cf_schedule import flags,validate_schedule


class FocusedExperiment(NetworkExperiment):
    def one(self,profile,arm,phase,ordinal,mode,hit_fraction=None,arm_reset_ms=0.):
        role='GATE' if phase=='gate' else 'AUXILIARY' if phase.startswith('aux_') else 'PRIMARY'
        before=len(self.network_rows)
        try:
            row=super().one(profile,arm,phase,ordinal,mode,hit_fraction,arm_reset_ms)
        except BaseException as exc:
            row=self.network_rows[-1] if len(self.network_rows)>before else dict(status='FAILED',request_id='NOT_STARTED',
                condition_id=profile['id'],arm=arm,phase=phase,ordinal=ordinal,mode=mode,reason=f'{type(exc).__name__}: {exc}')
            append(self.output/'r3cf_requests.jsonl',dict(row,**flags(self.fixture),analysis_role=role,
                schedule_sha256=self.plan['schedule_manifest']['schedule_sha256']))
            raise
        focused=dict(row,**flags(self.fixture),analysis_role=role,
            schedule_sha256=self.plan['schedule_manifest']['schedule_sha256'])
        append(self.output/'r3cf_requests.jsonl',focused);return focused

    def run_grid(self):
        validate_schedule(self.plan['schedule_manifest'])
        for profile in self.plan['profiles']:
            cid=profile['id']
            control(self.proxy['control_port'],self.token,'PROFILE',condition=profile,
                expected=dict(es_to_ud=self.channel.sent,ud_to_es=self.channel.received))
            self.condition_gates[cid]=dict(status='RUNNING',started_monotonic_ns=time.monotonic_ns())
            write(self.output/'r3c_condition_gates.json',self.condition_gates)
            for i,mode in enumerate(('NO_REUSE','COLD_MISS','WARM_HIT')):self.one(profile,'GATE','gate',i,mode)
            self.condition_gates[cid]=dict(status='PASS',completed_monotonic_ns=time.monotonic_ns(),
                cache_no_reuse_bitwise_equal=True,native_parity_separate=True)
            write(self.output/'r3c_condition_gates.json',self.condition_gates)
            for task in (t for t in self.plan['schedule'] if t['condition']['id']==cid):
                start=time.monotonic_ns();self.cache.reset('focused block precondition outside request wall')
                if not self.fixture:
                    import torch
                    torch.cuda.synchronize(self.device)
                reset=(time.monotonic_ns()-start)/1e6
                append(self.output/'r3cf_preconditions.jsonl',dict(condition_id=cid,arm=task['arm'],role=task['role'],
                    reset_ms=reset,cache_event=self.cache.events[-1],cache_bytes_after=self.cache.bytes,monotonic_ns=time.monotonic_ns()))
                for i,mode in enumerate(task['warmup_modes']):
                    self.one(profile,task['arm'],task['warmup_phase'],i,mode,task['hit_fraction'],reset if i==0 else 0.)
                for i,mode in enumerate(task['measured_modes']):
                    self.one(profile,task['arm'],task['measured_phase'],i,mode,task['hit_fraction'])
            print('R3C-F '+cid+' COMPLETE',flush=True)


def execute(plan,output):
    # The old API has no experiment-class argument. Scope this binding to ONE
    # isolated new 101 worker, then restore in finally. No production source,
    # Channel, CPU math, tensor path or timing implementation changes.
    from . import r3c_runtime
    original=r3c_runtime.NetworkExperiment
    try:
        r3c_runtime.NetworkExperiment=FocusedExperiment
        result=r3c_runtime.execute(plan,output)
        result.update(flags(plan['fixture']));return result
    finally:r3c_runtime.NetworkExperiment=original
