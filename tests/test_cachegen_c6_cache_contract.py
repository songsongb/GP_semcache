"""No-model tests against the installed GlobalCache implementation, never a replacement."""
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.cache.cache_entry import CacheEntry
from semcache.experiments.cachegen import c6_runtime as runtime
from semcache.semantic.subsequence import Subsequence

EXTENDED = 'physical_codec' in inspect.signature(runtime.GlobalCache).parameters


class TensorStub:
    """Only entry ownership/accounting interfaces; no numerical codec claim."""
    dtype = 'float16'
    device = 'cpu'
    def __init__(self, size=24):
        self.size = size
    def detach(self): return self
    def untyped_storage(self): return self
    def nbytes(self): return self.size
    def data_ptr(self): return id(self)


class View:
    def __init__(self, resident):
        self.resident = resident
        self.tensors = {0:(resident.q_tensors[0], TensorStub(), TensorStub())}
    def __getattr__(self, name): return getattr(self.resident, name)


class CodecSpy:
    def __init__(self): self.decodes = 0
    def decode_entry(self, resident):
        self.decodes += 1
        return View(resident)


def episode():
    return dict(cluster=0,token_ids=[10,11,12],source_start=3,target_start=4,
                source_user='user_a',source_id='source')


def entry(compressed, frame_bytes=7, ids=(10,11,12)):
    metadata=dict(component_scope='total_qkv',source_user='user_a',source_id='source')
    kwargs = (dict(q_tensors={0:TensorStub()},compressed_kv=SimpleNamespace(bitstream=b'x'*frame_bytes))
              if compressed else dict(tensors={0:tuple(TensorStub() for _ in range(3))}))
    return CacheEntry(0,ids,(3,6),runtime.RAW_ENTRY_BYTES,qkv_metadata=metadata,**kwargs)


class RawTests(unittest.TestCase):
    def test_raw_and_transport_only_keep_raw_storage(self):
        for mode in ('RAW_SEMCACHE','TRANSPORT_QKV_COMP'):
            cache=runtime.make_c6_cache(mode)
            resident=entry(False)
            hit=runtime.insert_and_lookup_c6(cache,resident,episode(),Subsequence((10,11,12),4,7))
            self.assertIs(hit.entry,resident)
            self.assertEqual(cache.logical_cache_bytes,runtime.RAW_ENTRY_BYTES)
            self.assertIsNone(getattr(cache,'physical_codec',None))
            self.assertEqual(resident.frequency,1)

    def test_full_recompute_has_no_cache(self):
        with self.assertRaisesRegex(ValueError,'must not create'):
            runtime.make_c6_cache('FULL_RECOMPUTE')


@unittest.skipUnless(EXTENDED, 'Current checkout lacks the C2 GlobalCache API')
class CompressedContractTests(unittest.TestCase):
    def storage(self):
        # Production obtains this value from C2 MODE_COMPRESSED, not a C6 literal.
        return SimpleNamespace(mode='COMPRESSED_KV_K20_V16',codec=CodecSpy(),
            validate_decoded=lambda resident,view: self.assertIs(view.resident,resident))

    def test_registered_codec_and_exactly_one_decode(self):
        for mode in ('STORAGE_KV_COMP','FULL_PIPELINE'):
            storage=self.storage()
            cache=runtime.make_c6_cache(mode,storage)
            self.assertIs(cache.physical_codec,storage.codec)
            self.assertEqual(cache.physical_storage_mode,storage.mode)
            self.assertIsNone(cache.capacity_charge)
            self.assertEqual(cache.shared_overhead_bytes,0)
            resident=entry(True)
            original_q=resident.q_tensors[0]
            hit=runtime.insert_and_lookup_c6(cache,resident,episode(),Subsequence((10,11,12),4,7),storage)
            self.assertEqual(storage.codec.decodes,1)
            self.assertIs(cache.entries[resident.key],resident)
            self.assertIsNone(resident.tensors)
            self.assertIs(hit.entry.resident,resident)
            self.assertIsNot(hit.entry,resident)
            self.assertEqual(len(hit.entry.tensors[0]),3)
            self.assertIs(hit.entry.tensors[0][0],original_q)
            self.assertEqual(original_q.dtype,'float16')
            self.assertEqual(cache.charged_cache_bytes,runtime.RAW_ENTRY_BYTES)
            self.assertEqual(cache.physical_tensor_bytes,24+7)
            self.assertEqual(resident.frequency,1)

    def test_physical_size_cannot_change_admission_eviction_or_hits(self):
        traces=[]
        for compressed,frame_bytes in ((False,0),(True,1),(True,runtime.RAW_ENTRY_BYTES+100)):
            cache=runtime.make_c6_cache('STORAGE_KV_COMP',self.storage()) if compressed else runtime.make_c6_cache('RAW_SEMCACHE')
            first=entry(compressed,frame_bytes)
            second=entry(compressed,frame_bytes,ids=(20,21,22))
            trace=[]
            events=[]
            for value in (first,second):
                trace.append(cache.insert(value,on_event=lambda kind,e,score: events.append((kind,e.key))))
                self.assertEqual(cache.charged_entry_bytes(value),runtime.RAW_ENTRY_BYTES)
                self.assertEqual(cache.charged_cache_bytes,runtime.RAW_ENTRY_BYTES)
                self.assertEqual(cache.logical_cache_bytes,runtime.RAW_ENTRY_BYTES)
            traces.append((trace,events,sorted(cache.entries),cache.lookup(second.key) is not None))
        self.assertEqual(traces[0],traces[1])
        self.assertEqual(traces[0],traces[2])
        self.assertTrue(any(kind=='EVICT' for kind,key in traces[0][1]))

    def test_global_cache_guard_remains_active(self):
        cache=runtime.make_c6_cache('RAW_SEMCACHE')
        with self.assertRaisesRegex(ValueError,'Compressed entry requires a compressed cache codec'):
            cache.insert(entry(True))
        self.assertEqual(cache.entries,{})

    def test_compressed_entry_rejects_raw_kv(self):
        with self.assertRaisesRegex(ValueError,'cannot retain raw K/V'):
            CacheEntry(0,(10,11,12),(3,6),runtime.RAW_ENTRY_BYTES,
                tensors={0:tuple(TensorStub() for _ in range(3))},
                q_tensors={0:TensorStub()},compressed_kv=SimpleNamespace(bitstream=b'kv'))

    def test_c6_rejects_changed_logical_size(self):
        storage=self.storage()
        resident=entry(True)
        resident.size_bytes=7
        with self.assertRaisesRegex(ValueError,'fixed raw logical size'):
            runtime.insert_and_lookup_c6(runtime.make_c6_cache('FULL_PIPELINE',storage),resident,
                episode(),Subsequence((10,11,12),4,7),storage)
        self.assertEqual(storage.codec.decodes,0)


if __name__ == '__main__': unittest.main()
