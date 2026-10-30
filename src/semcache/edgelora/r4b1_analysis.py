"""Separate reuse fidelity, native numerical parity and diagnostic latency."""
from .r3a_artifacts import read,write
from .r4a_analysis import analyze as analyze_full
from .r4a_quality import evaluate
from .r4b1_evidence import flags


def analyze(es,prepared,rows,tokens,worker,*,fixture,gold=None):
    contrasts,original=analyze_full(es,prepared,rows,tokens,worker,fixture=fixture)
    native=None if fixture else read(prepared/'native_index.json')['generations']
    quality,signatures=evaluate(rows,native,gold)
    decision=dict(execution_status='COMPLETE',R4B1_PHYSICAL_E2E_PASS=not fixture,
        R4B1_MULTI_ENTRY_RESIDENCY_PASS=not fixture,R4B1_RAW_QKV_E2E_FIDELITY_PASS=not fixture,
        fixture_checks_passed=fixture,NATIVE_PEFT_STRICT_PARITY_PASS=original['NATIVE_PEFT_STRICT_PARITY_PASS'],
        native_equivalent=original['native_equivalent'],native_generation_checks=original['native_generation_checks'],
        by_cell=original['by_cell'],latency_contrasts=contrasts,
        qualification_note='B1 diagnostic only; original Native parity remains a separate strict gate; no primary or correctness-qualified speedup claim',
        **flags(fixture))
    quality.update(native_peft_strict_parity_passed=decision['NATIVE_PEFT_STRICT_PARITY_PASS'])
    return decision,quality,signatures


def export(output,decision,quality,signatures):
    write(output/'r4b1_decision.json',decision);write(output/'r4b1_quality.json',quality);write(output/'r4b1_sacrebleu_signatures.json',signatures)
    from .r3a_artifacts import write_csv
    write_csv(output/'r4b1_latency_contrasts.csv',[{k:c[k] for k in ('cell_id','mode','metric','delta_mean_ms','ci95_low_ms','ci95_high_ms','pairs','repeat_blocks','benefit_observed')}
        for c in decision.get('latency_contrasts',[])])
