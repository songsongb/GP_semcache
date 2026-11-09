"""Arithmetic contracts for LAYER_INPUT_BYTES_VERIFIED_V1; no GPU entry point.

Native arithmetic is supported ONLY with a paid full-shape output-byte guard.
Row skipping is limited to the explicitly synthetic scalar fixture contract.
"""
from copy import deepcopy
import math
from .communication import require, digest
from .r3a_artifacts import unpack
from .r3a_tiny import linear, f32
from .r3b_fixture import ArrayTensor
from .r5_sparse_qkv import blob, SparseSession
from .r5_subsequence_physical import tensor_proof

REFERENCE = 'FULL_SHAPE_REFERENCE_OUTPUT_GUARD_V1'
SCALAR = 'TEST_FIXTURE_ONLY_ROW_LOCAL_F32_V1'


def finite(t):
    dtype = str(t.dtype).removeprefix('torch.')
    require(dtype in ('float16', 'float32'), 'unsupported projection dtype')
    require(all(math.isfinite(v) for v in unpack(blob(t), dtype)), 'nonfinite projection tensor')


class ScalarFixtureArithmetic:
    """Fixed row-local scalar Linear + existing scalar CPU LoRA, not OPT/GEMM.

    shape_offset is an explicit correctness fixture modeling shape-dependent
    arithmetic. It is allowed only with the full-shape output guard.
    """
    def __init__(self, identity, weights, matrices, *, reference_guard=False, shape_offset=False, biases=None):
        require(identity['device'] == 'cpu' and identity['dtype'] == 'float32', 'fixture CPU FP32 required')
        require(not shape_offset or reference_guard, 'shape-dependent fixture requires output guard')
        self.identity = deepcopy(identity); self.weights = deepcopy(weights); self.matrices = deepcopy(matrices)
        self.biases = deepcopy(biases or {(l, r): [0.]*identity['hidden_size'] for l in range(identity['layers']) for r in 'qkv'})
        self.reference_guard = reference_guard; self.shape_offset = shape_offset
        self.kind = REFERENCE if reference_guard else SCALAR
        self._functions = {}
        for l in range(identity['layers']):
            for r in 'qkv':
                def native(x, l=l, r=r):
                    h = self.identity['hidden_size']; data = unpack(blob(x), 'float32')
                    values = linear(data, self.weights[l, r], h, h)
                    return ArrayTensor([f32(v+self.biases[l, r][n % h]+(x.shape[1] if self.shape_offset else 0))
                                        for n, v in enumerate(values)], x.shape)
                self._functions[l, r] = native
        self.binding = self.snapshot()

    def snapshot(self):
        return digest(dict(kind=self.kind, identity=self.identity, weights=sorted(self.weights.items()),
                           matrices=sorted(self.matrices.items()), biases=sorted(self.biases.items()),
                           shape_offset=self.shape_offset, algorithm='canonical FP32 input; fixed Python row dot; f32 Linear; f32 bias; scalar cpu_deltas; f32 add'))

    def native(self, layer, role): return self._functions[layer, role]

    def check(self, req, layer, role, x, native, session):
        require(req['TEST_FIXTURE_ONLY'] and type(x) is ArrayTensor and x.dtype == 'float32' and x.device == 'cpu',
                'scalar contract is TEST_FIXTURE_ONLY, never native inference')
        require(self.snapshot() == self.binding and req['physical_identity'] == self.identity, 'fixture weights/adapter/backend binding changed')
        require(native is self._functions[layer, role], 'unbound fixture projection function')
        require(getattr(session, 'arithmetic_binding', None) == self.binding, 'fixture CPU LoRA binding differs')
        if hasattr(session, 'matrices'):
            require(session.matrices == self.matrices, 'fixture CPU LoRA weights changed')
        require(x.is_contiguous() and x.shape == (1, len(req['token_ids']), self.identity['hidden_size']), 'unsupported fixture shape/layout')
        finite(x)

    def metadata(self):
        return dict(contract=self.kind, binding=self.binding, TEST_FIXTURE_ONLY=True,
                    reference_guard=self.reference_guard, sparse_row_equivalence='fixed row loops only' if not self.reference_guard else 'NOT_ASSUMED')


class NativeReferenceArithmetic:
    """Pinned bare Linear and real sparse UD, with all rows recomputed.

    This establishes equality to the actual current full-shape projection output,
    not equality of independently evaluated GPU models or sparse GEMM kernels.
    Construction fingerprints existing weights and can incur host transfers.
    The CPU CLI never constructs this contract on a GPU or loads a GPU model.
    """
    kind = REFERENCE
    reference_guard = True

    def __init__(self, identity, adapter):
        import torch
        self.identity = deepcopy(identity); self.adapter = adapter
        require(len(adapter.layers) == identity['layers'] and identity['rank'] == 8, 'native architecture/rank differs')
        self.modules = {(l, r): m for l in range(len(adapter.layers)) for r, m in adapter.projection_modules(l).items()}
        require(set(self.modules) == {(l, r) for l in range(identity['layers']) for r in 'qkv'}, 'native QKV coverage differs')
        for m in self.modules.values():
            require(type(m) is torch.nn.Linear and not m.training and not m._forward_hooks and not m._forward_pre_hooks,
                    'bare eval Linear required')
        from .r5_i5r2b_capture import backend
        self.backend_binding = digest(backend())
        self.projection_proofs = {key: self.module_proof(m) for key, m in self.modules.items()}
        self.binding = digest(dict(contract=self.kind, identity=self.identity, backend=self.backend_binding,
                                   scale=1., rank=8, projections=sorted(self.projection_proofs.items())))
        self.ud_binding = None

    @staticmethod
    def module_proof(m):
        return dict(weight=tensor_proof(m.weight), bias=None if m.bias is None else tensor_proof(m.bias))

    def check(self, req, layer, role, x, native, session):
        import torch
        from .r5_i5r2b_capture import backend
        require(not isinstance(x, ArrayTensor) and req['physical_identity'] == self.identity, 'native physical identity differs')
        require(not torch.is_autocast_enabled() and x.is_contiguous() and str(x.dtype) == 'torch.'+self.identity['dtype'] and
                str(x.device) == self.identity['device'] and list(x.shape) == [1, len(req['token_ids']), self.identity['hidden_size']],
                'unsupported native dtype/device/shape/layout/autocast')
        module = self.modules[layer, role]
        h = self.identity['hidden_size']
        require(tuple(module.weight.shape) == (h,h) and module.weight.is_contiguous() and
                module.weight.dtype == x.dtype and module.weight.device == x.device and
                (module.bias is None or tuple(module.bias.shape) == (h,) and module.bias.is_contiguous() and
                 module.bias.dtype == x.dtype and module.bias.device == x.device) and
                not module._forward_hooks and not module._forward_pre_hooks, 'unsupported native projection weights/bias/layout')
        require(getattr(native, '__self__', None) is module and getattr(native, '__func__', None) is torch.nn.Linear.forward,
                'native projection callable differs')
        require(not module.training and self.module_proof(module) == self.projection_proofs[layer, role] and
                digest(backend()) == self.backend_binding, 'native weights/bias/backend contract changed')
        require(type(session) is SparseSession and session.req == req and not session.fixture and
                session.channel.adapter == self.identity['adapter_matrix_fingerprint'], 'native CPU LoRA channel binding differs')
        ud_binding = digest(dict(session=session.channel.session, adapter=session.channel.adapter))
        if self.ud_binding is None: self.ud_binding = ud_binding
        require(self.ud_binding == ud_binding, 'native CPU UD session changed inside owner lifetime')
        if session.receipts and session.receipts[-1]['context']['layer'] == layer:
            receipt = session.receipts[-1]
            require(receipt['context']['fresh_positions'] == list(range(len(req['token_ids']))), 'reference guard requires full-shape CPU LoRA')
            precision = receipt['received']['header']['metadata']['actual_precision']
            for r in 'qkv':
                require(precision[r]['scale'] == 1. and all(precision[r][k] == 'torch.float32' for k in
                    ('input_dtype','A_weight_dtype','A_output_dtype','B_weight_dtype','B_output_dtype')), 'CPU LoRA arithmetic precision/scale differs')
        finite(x)

    def metadata(self):
        return dict(contract=self.kind, binding=self.binding, TEST_FIXTURE_ONLY=False, reference_guard=True,
                    ud_session_binding=self.ud_binding,
                    sparse_row_equivalence='NOT_ESTABLISHED; all target rows recomputed and output bytes checked')
