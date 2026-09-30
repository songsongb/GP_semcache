"""Dependency-free routing checks; these do not validate real CacheGen codecs."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

PATH = Path(__file__).resolve().parents[1] / 'scripts/48_validate_cachegen_hit_miss_pipeline.py'
spec = importlib.util.spec_from_file_location('cachegen_pipeline_smoke', PATH)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class RoutingTests(unittest.TestCase):
    def test_miss_then_hit_uses_current_q_and_skips_transport(self):
        transport = Mock(calls=0)
        initial = {0: (object(), object(), object())}
        current_q = {0: object()}
        payload = b'compressed-kv-test-double'

        def transmit(tensors):
            transport.calls += 1
            return tensors

        transport.transmit.side_effect = transmit
        storage = Mock()
        storage.encode.return_value = payload
        storage.decode.side_effect = lambda resident, q: {0: (q[0], initial[0][1], initial[0][2])}
        project = Mock(side_effect=lambda q_only: current_q if q_only else initial)
        downstream = Mock()
        pipeline = smoke.Pipeline(transport, storage, project, downstream)
        miss, _ = pipeline.request('same request')
        # Any accidental communication on HIT must fail immediately.
        transport.transmit.side_effect = AssertionError('HIT used network')
        hit, restored = pipeline.request('same request')
        self.assertEqual(miss, dict(cache_lookup='MISS', network_path_used=True,
                                   cache_inserted=True, stored_q=False, stored_k=True, stored_v=True))
        self.assertEqual(hit, dict(cache_lookup='HIT', network_path_used=False,
                                  storage_decode_used=True, q_recomputed_or_current=True))
        self.assertEqual(pipeline.cache, {'same request': payload})
        self.assertIs(restored[0][0], current_q[0])
        storage.encode.assert_called_once_with(initial)
        storage.decode.assert_called_once_with(payload, current_q)
        self.assertEqual(downstream.call_count, 2)
        self.assertEqual([call.kwargs for call in project.call_args_list],
                         [{'q_only': False}, {'q_only': True}])

    def test_storage_adapter_discards_q_at_resident_boundary(self):
        adapter = smoke.StorageCacheGenCodec.__new__(smoke.StorageCacheGenCodec)
        adapter.encodes = adapter.decodes = 0
        payload = object()
        codec = Mock()
        codec.make_entry.side_effect = lambda entry_type, *args: entry_type(
            *args[:3], compressed_kv=payload, q_tensors={'must_not_persist': object()})
        adapter.codec = codec
        resident = adapter.encode({})
        self.assertIs(resident, payload)
        q = {0: object()}
        adapter.decode(resident, q)
        view = codec.decode_entry.call_args.args[0]
        self.assertIs(view.compressed_kv, payload)
        self.assertIs(view.q_tensors, q)


if __name__ == '__main__':
    unittest.main()
