"""M6 quantities always carry a source and scope; no inferred measurement source."""
from enum import Enum
import math


class Provenance(str, Enum):
    PAPER_DEFINED = 'PAPER_DEFINED'
    REPRODUCTION_CHOICE = 'REPRODUCTION_CHOICE'
    MEASURED = 'MEASURED'
    SIMULATED = 'SIMULATED'
    PAPER_REFERENCE = 'PAPER_REFERENCE'


def metric(value, source, scope, unit):
    source = Provenance(source)
    if not scope or not unit:
        raise ValueError('Metric scope and unit are required')
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
        raise ValueError('Metric value must be finite numeric or null')
    return dict(value=value, metric_source=source.value, metric_scope=scope, unit=unit)


def configuration_provenance(config):
    """Per-leaf source ledger. Unknown/overridden settings are choices, not paper facts."""
    paper = {
        'num_users':50,'lora_rank':8,'subsequence_window':3,'logical_cache_capacity_gb':20,
        'clustering.update_mode':'immediate_eq9',
        'clustering.update_rule':'incremental_mean_after_every_assignment',
        'admission.alpha':.5,'admission.beta':.3,
        'admission.delta':.2,'admission.threshold':.3,'eviction.alpha':.4,'eviction.beta':.3,
        'eviction.gamma':.2,'eviction.delta':.1,'semantic_impact.rho':.8,'semantic_impact.history_lambda':100,
        'system.es_gpu':'NVIDIA A100 80GB','system.es_cpu':'Intel Xeon Gold 6338',
        'system.ud_cpu_cores':4,'system.ud_ghz':2.3,'system.ud_ram_gb':8,'system.bandwidth_mbps':200,
        'semantic_encoder.family':'TinyBERT','lora.implementation':'Hugging Face PEFT',
        'lora.target_matrices':['Q','K','V'],
        'cluster_count':{'multiwoz':20,'coqa':40,'snips':30}.get(config.get('dataset'))}
    result = {}
    def visit(value, prefix=''):
        for key,item in value.items():
            path = f'{prefix}.{key}' if prefix else key
            if key == 'provenance':
                continue
            if isinstance(item,dict):
                visit(item,path)
            else:
                result[path] = 'PAPER_DEFINED' if path in paper and item == paper[path] and item is not None else 'REPRODUCTION_CHOICE'
    visit(config)
    return result
