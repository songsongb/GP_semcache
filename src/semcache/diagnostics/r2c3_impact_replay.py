"""Scoped exact-value impact replay, with post-request deterministic policy proof.

No production function is changed. CPU snapshots/hashes live outside the frozen
Engine request timer. The reduction dispatch counter, CPU mask/attention metadata
observation and Replay's candidate-order/metadata checks execute inside it.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from collections import Counter, deque
import math
from unittest.mock import patch

from semcache.metrics.r1 import digest
from semcache.diagnostics.r1b3_break_even import MissingEvidence, ParityFailure

ARMS = ('FULL_NATIVE', 'SEMCACHE_NO_REUSE_IMPACT_ON',
        'SEMCACHE_NO_REUSE_IMPACT_REPLAY', 'RAW_SEMCACHE_QKV_IMPACT_ON',
        'RAW_SEMCACHE_QKV_IMPACT_REPLAY')
BASES = ('SEMCACHE_NO_REUSE', 'RAW_SEMCACHE_QKV')
SETTINGS = dict(use_cache=False,output_attentions=True,attention_backend='eager',head='native_all_token_logits',
    impact_execution='token_precompute',adapter='user_a',atol=1e-3,rtol=1e-3)
POLICY_EVENTS = frozenset(('CLUSTER_ASSIGN', 'CACHE_LOOKUP', 'HIT', 'MISS',
    'NON_OVERLAP_RESOLVE', 'REUSE_PROVENANCE', 'CHU', 'ADMISSION_SCORE', 'ADMIT',
    'DENY', 'DEDUPLICATE', 'INSERT', 'EVICT', 'PBR', 'PBR_TRIGGER'))


def canonical(value):
    """Exact Python float bits, typed mapping keys; no GPU tensor conversion."""
    if value is None or isinstance(value, (str, bool, int)): return value
    if isinstance(value, float):
        if not math.isfinite(value): raise MissingEvidence('nonfinite policy/impact value')
        return {'float_hex': value.hex()}
    if isinstance(value, dict):
        pairs = [(canonical(k), canonical(v)) for k,v in value.items()]
        return {'mapping': sorted(pairs, key=lambda p: digest(p[0]))}
    if isinstance(value, (tuple,list,deque)):
        return [canonical(x) for x in value]
    raise MissingEvidence('unsupported policy metadata: '+type(value).__name__)


def entry_state(entry):
    # Logical query clocks are policy state, NOT wall timestamps to discard.
    return {name:getattr(entry,name) for name in ('cluster_id','token_ids','positions',
        'size_bytes','qkv_metadata','frequency','impact','created_at','last_access','updated_at')}


def policy_state(engine, catalog):
    cache = engine.cache
    return canonical(dict(cache=dict(capacity=cache.capacity_bytes, now=cache.now,
        hits=cache.hits, misses=cache.misses, peak_logical_bytes=cache.peak_logical_cache_bytes,
        entry_order=list(cache.entries),
        entries={k:entry_state(v) for k,v in cache.entries.items()},
        admission=vars(cache.admission), eviction=vars(cache.eviction)),
        appearances=engine.metrics.appearances, frequencies=engine.metrics.frequencies,
        appearance_window=engine.metrics.appearances.maxlen,
        history=dict(engine.metrics.history.history), history_lambda=engine.metrics.history.history_lambda,
        updater={k:v for k,v in vars(engine.metrics.updater).items() if k!='provider'},
        clusterer=vars(engine.clusterer), query_id=engine.query_id,
        admission_candidate_order=engine._admission_candidate_order, admission_order=engine._admission_order,
        catalog_entry_order=list(catalog.entries),
        catalog_entries={k:entry_state(v) for k,v in catalog.entries.items()}))


def policy_proof(before, after, result):
    events = [e for e in result['events'] if e['event_type'] in POLICY_EVENTS]
    summary = result['summary']
    inputs = {k:summary[k] for k in ('token_ids','cluster_id','impact_windows',
        'admission_audit','deduplication_audit','chu_update_count','pbr_updates_this_query')}
    return dict(before_digest=digest(before), after_digest=digest(after),
        inputs_digest=digest(canonical(inputs)), events_digest=digest(canonical(events)),
        event_counts=dict(Counter(e['event_type'] for e in events)),
        admitted=summary['admitted_block_count'], rejected=summary['rejected_block_count'],
        history_and_catalog_included=True,
        excluded='wall timestamps/timing scopes only; logical query clocks retained')


def window_sequence(windows):
    return tuple((w.start,w.end,tuple(w.token_ids)) for w in windows)


def attention_metadata(attentions):
    if not attentions or any(t is None for t in attentions):
        raise MissingEvidence('normal eager attention outputs required')
    return tuple((tuple(t.shape), str(t.dtype), str(t.device)) for t in attentions)


def mask_values(mask):
    if mask is None or getattr(mask,'is_cuda',True):
        raise MissingEvidence('expected existing CPU valid-attention mask')
    return tuple(mask.tolist())


@dataclass(frozen=True)
class ImpactReference:
    binding_digest: str
    windows: tuple
    values: tuple
    attention: tuple
    mask: tuple
    policy_digest: str
    sha256: str

    @classmethod
    def capture(cls, binding, session):
        if session.real_reduction_calls != 1 or session.dispatch_calls != 1 or session.fallback_calls:
            raise MissingEvidence('reference must come from one actual TOKEN_PRECOMPUTE request')
        data = dict(binding_digest=digest(binding), windows=window_sequence(session.windows),
            values=tuple(session.values), attention=session.attention, mask=session.mask,
            policy_digest=digest(session.policy))
        if len(data['windows']) != len(data['values']) or not data['windows']:
            raise MissingEvidence('reference missing candidate values')
        if any(not isinstance(v,float) or not math.isfinite(v) or v<0 for v in data['values']):
            raise MissingEvidence('invalid actual impact values')
        return cls(**data, sha256=digest(canonical(data)))

    def verify(self, binding, expected_digest):
        data = {k:getattr(self,k) for k in ('binding_digest','windows','values','attention','mask','policy_digest')}
        if (not expected_digest or self.sha256 != expected_digest or
            self.sha256 != digest(canonical(data)) or self.binding_digest != digest(binding)):
            raise MissingEvidence('Replay reference/provenance/digest changed')

    def export(self, binding):
        return dict(provenance=binding, digest=self.sha256, binding_digest=self.binding_digest,
            candidate_windows=[dict(start=a,end=b,token_ids=list(ids),value=v,value_hex=v.hex())
                for (a,b,ids),v in zip(self.windows,self.values)],
            attention_metadata=self.attention, valid_mask=self.mask, policy_digest=self.policy_digest,
            origin='untimed actual TOKEN_PRECOMPUTE request, independently captured per exact base mode',
            classification='ORACLE_DIAGNOSTIC', ablation='ORACLE_UPPER_BOUND',deployable_predictor=False)


class ImpactSession:
    def __init__(self, runtime, binding, reference=None, expected_digest=None):
        self.runtime=runtime; self.binding=binding; self.reference=reference
        if reference is not None: reference.verify(binding,expected_digest)
        self.dispatch_calls=self.real_reduction_calls=self.fallback_calls=0
        self.engine_calls=0; self.policy=None; self.windows=self.values=None

    def finish(self):
        if self.engine_calls!=1 or self.dispatch_calls!=1 or self.fallback_calls or self.policy is None:
            raise MissingEvidence('one complete Engine request and one impact dispatch required')
        if self.real_reduction_calls != (0 if self.reference else 1):
            raise ParityFailure('Replay real impact reduction call count is not zero')
        if self.reference:
            if self.reference.policy_digest != digest(self.policy):
                error=ParityFailure('Replay deterministic policy inputs/history/CHU/Admission/state differ')
                error.evidence=dict(expected_digest=self.reference.policy_digest,actual_digest=digest(self.policy),
                    actual_policy=self.policy,expected_policy=getattr(self.runtime,'reference_policies',{}).get(self.binding['base_mode']))
                raise error

    @contextmanager
    def scoped(self):
        import semcache.metrics.r1_runtime as rt
        import semcache.cache.impact_execution as impact
        import semcache.semcache_engine as em
        import semcache.cache.metric_manager as metrics
        from semcache.cache.attention_impact import PaperRowL2SumReducer
        original_compute=impact.compute_window_values
        original_request=rt.engine_request
        def compute(attentions,windows,mode,valid_attention_mask=None,**kwargs):
            self.dispatch_calls+=1
            if self.dispatch_calls!=1 or mode!='TOKEN_PRECOMPUTE' or kwargs.get('production_reference'):
                raise MissingEvidence('unexpected impact dispatch/algorithm')
            # CPU metadata only, in both arms. Never retain attention tensors
            # beyond their original Engine lifetime or copy their contents.
            self.attention=attention_metadata(attentions)
            self.mask=mask_values(valid_attention_mask)
            if self.reference:
                # Metadata only: never detach/copy/materialize attention values.
                if (window_sequence(windows)!=self.reference.windows or
                    self.attention!=self.reference.attention or self.mask!=self.reference.mask):
                    raise MissingEvidence('Replay candidate order/length/tokens/mask/attention metadata mismatch')
                values=self.reference.values
            else:
                self.real_reduction_calls+=1
                values=original_compute(attentions,windows,mode,valid_attention_mask,**kwargs)
            self.windows=windows; self.values=values
            return values
        def forbidden(*args,**kwargs):
            self.fallback_calls+=1; self.real_reduction_calls+=1
            raise ParityFailure('unexpected real/fallback Attention Impact reduction')
        def request(engine,*args,**kwargs):
            self.engine_calls+=1
            if self.engine_calls!=1 or engine.impact_execution!='token_precompute' or not engine.capture_impact_values:
                raise MissingEvidence('C1 impact configuration/capture changed')
            before=policy_state(engine,self.runtime.catalog)  # outside Engine clock
            result=original_request(engine,*args,**kwargs)
            after=policy_state(engine,self.runtime.catalog)   # clock already closed
            summary=result['summary']
            expected=[(a,b,tuple(ids),float(v).hex()) for (a,b,ids),v in zip(window_sequence(self.windows),self.values)]
            actual=[(w['start'],w['end'],tuple(w['token_ids']),float(w['value']).hex()) for w in summary['impact_windows']]
            if actual!=expected or summary['token_ids']!=self.binding['token_ids']:
                raise ParityFailure('computed/replayed impact inputs or executed Token IDs differ')
            self.policy=policy_proof(before,after,result)
            self.impact_wall_ms=summary['impact_wall_ms']
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(rt,'engine_request',request))
            stack.enter_context(patch.object(impact,'compute_window_values',compute))
            if self.reference: stack.enter_context(patch.object(impact,'_validate',forbidden))
            stack.enter_context(patch.object(em,'actual_attention_impact',forbidden))
            stack.enter_context(patch.object(metrics,'actual_attention_impact',forbidden))
            stack.enter_context(patch.object(PaperRowL2SumReducer,'reduce',forbidden))
            try:
                yield self
                self.finish()
            except BaseException as exc:
                exc.impact_audit=dict(replay=self.reference is not None,dispatch_calls=self.dispatch_calls,
                    real_reduction_calls=self.real_reduction_calls,fallback_calls=self.fallback_calls)
                raise
