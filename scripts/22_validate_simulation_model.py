"""Hand-checkable analytical fixtures; no measured system speedup claim."""
import math
from _paper_common import parser, load_paper_config
from semcache.simulation.cost_model import projection_savings, paper_latency, communication_seconds, mbps_to_bits_per_second, tflops_to_flops_per_second
from semcache.experiments.manifest import write_manifest, run_manifest, canonical


def main():
    p=parser(__doc__); p.add_argument('--output',default='results/manifests/simulation_validation.json')
    a=p.parse_args(); c=load_paper_config(a.config)
    s=projection_savings(212,4096,8)
    assert s==dict(base_flops_saved=21340618752,lora_flops_saved=41680896,comm_elements_saved=3473408)
    assert communication_seconds(25_000_000,1,200)==1
    assert mbps_to_bits_per_second(200)==200_000_000
    assert tflops_to_flops_per_second(1)==1_000_000_000_000
    speedup=512/300
    assert math.isclose(speedup,1.7066666666666668)
    args=dict(n=512,d=4096,r=8,f_ES=1e12,f_UD=1e9,B=200e6,element_size_bytes=2)
    baseline=paper_latency(n_reused=0,**args); reused=paper_latency(n_reused=212,**args)
    assert reused['remaining_es_s']==baseline['remaining_es_s']
    result=dict(savings=s,simplified_speedup=speedup,baseline=baseline,reused=reused,
        assumptions=dict(fixture_provenance='REPRODUCTION_CHOICE',scope='one layer prefill',
        n=512,n_reused=212,d=4096,r=8,f_ES_flops_per_s=1e12,f_UD_flops_per_s=1e9,
        wire_bytes_per_element=2,bandwidth_mbps=200),metric_source='SIMULATED',measured_speedup_claimed=False)
    write_manifest(a.output,run_manifest('simulation_validation',c,None,validation=result))
    print(canonical(result))


if __name__=='__main__': main()
