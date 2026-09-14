"""Read-only comparison registry. Never imported by simulator/runner/config.

Memory is paper system memory in GB, not CUDA allocation. Values transcribed
from supplied Table II/Fig.6; no executable model, tuning or generated results.
"""
from types import MappingProxyType


def _freeze(value):
    return MappingProxyType({k:_freeze(v) for k,v in value.items()}) if isinstance(value,dict) else value


def _row(latency, memory, bleu, scope='paper Table II system evaluation'):
    return dict(latency_s=latency, system_memory_gb=memory, bleu_percent=bleu,
                metric_source='PAPER_REFERENCE', metric_scope=scope)

TABLE_II = _freeze({
 'GPT2':dict(UD_ONLY=_row(6.15,220.5,72.4), FBC=_row(3.57,23.9,71.4), SEMCACHE=_row(2.91,23.9,71.9), ES_ONLY=_row(2.54,24.5,72.4)),
 'LLaMA':dict(UD_ONLY=_row(7.53,270.1,72.9), FBC=_row(4.85,25.1,72.4), SEMCACHE=_row(3.64,25.1,72.5), ES_ONLY=_row(3.16,25.6,72.9)),
 'OPT':dict(UD_ONLY=None, FBC=_row(6.94,35.2,72.3), SEMCACHE=_row(6.07,35.2,73.1), ES_ONLY=_row(5.71,36.8,73.4))})
FIGURE_6 = _freeze(dict(metric_source='PAPER_REFERENCE',
 granularity={'Vanilla EdgeLoRA':_row(10.68,39.5,74.5,'paper Fig.6 granularity'),'ILC':_row(9.56,32.5,73.2,'paper Fig.6 granularity'),'ILC+TLS':_row(6.07,15.1,73.1,'paper Fig.6 granularity')},
 cache_metrics={name:dict(hit_rate_percent=hit,latency_s=lat,metric_source='PAPER_REFERENCE') for name,hit,lat in
 [('F',67.3,7.97),('F+A',71.4,7.45),('F+A+I',73.4,6.87),('F+A+I+S',76.7,6.07)]},
 impact_update={name:dict(hit_rate_percent=hit,latency_s=lat,metric_source='PAPER_REFERENCE') for name,hit,lat in
 [('CHU',71.1,6.74),('PBR',69.8,7.53),('CHU+PBR',76.7,6.07)]}))
