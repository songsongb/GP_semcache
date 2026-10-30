"""R4-B1 accounting adapter; the frozen R4-A forward/decode math is unchanged."""
from .communication import require
from .r4a_generation import generate as full_generation


def generate(driver,prompt,policy,request_id,router,mode):
    require(not driver.native,'R4-B1 measured endpoint must be distributed')
    before=router.snapshot();start=len(router.events);router.request_id=request_id
    row=full_generation(driver,prompt,policy,request_id,router=router,mode=mode)
    events=router.events[start:]
    row.update(cache_events=events,cache_before=before,cache_after=router.snapshot(),
        cache_entries_before=before['entry_count'],cache_entries_after=len(router.entries),
        cache_eviction_ms=sum(e['duration_ms'] for e in events if e['kind'] in ('EVICT','INVALIDATE','REJECT')),
        cache_insert_ms=sum(e['duration_ms'] for e in events if e['kind']=='ADMISSION'),
        cache_lookup_validation_ms=sum(e['duration_ms'] for e in events if e['kind'] in ('HIT','MISS')),
        cache_accounting_scope='owned compact QKV tensors across all entries; excludes model, in-request KV, captures and CUDA allocator reserve')
    return row
