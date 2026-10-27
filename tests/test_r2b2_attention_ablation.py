"""Dependency-free attention/endpoint contracts; real installed API tested separately."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from semcache.diagnostics.r2b2_attention_ablation import B2API,endpoint_policy,backend_audit,validate_attention_output
from semcache.diagnostics.r2b1_prefix_cache import PrefixError


class ArmTests(unittest.TestCase):
    def test_endpoint_capability_no_silent_custom_head(self):
        self.assertEqual(endpoint_policy([], 'auto')['effective'],'b1_all_logits')
        self.assertTrue(endpoint_policy([], 'auto')['materialization_confounded'])
        with self.assertRaises(PrefixError): endpoint_policy([], 'native-last')
        self.assertEqual(endpoint_policy(['logits_to_keep'],'auto')['native_option'],'logits_to_keep')
        self.assertEqual(endpoint_policy(['num_logits_to_keep'],'native-last')['native_option'],'num_logits_to_keep')
        self.assertIsNone(endpoint_policy(['logits_to_keep'],'b1')['native_option'])

    def test_both_arms_same_inputs_cache_positions_and_backend(self):
        original=NS(kwargs=lambda *a,**kw:dict(input_ids=[1,2],position_ids=[3,4],attention_mask=[1]*5,
            use_cache=True,return_dict=True,output_attentions=True,logits_to_keep=0))
        ep=endpoint_policy(['logits_to_keep'],'auto')
        on=B2API(original,'on',ep).kwargs(); off=B2API(original,'off',ep).kwargs()
        self.assertTrue(on.pop('output_attentions')); self.assertFalse(off.pop('output_attentions'))
        self.assertEqual(on,off); self.assertEqual(on['logits_to_keep'],1)
        self.assertTrue(on['use_cache'])

    def test_fallback_rejected_and_returned_output_validated(self):
        class OPTAttention:
            def forward(self): pass
        attention=OPTAttention(); config=NS(_attn_implementation='eager'); attention.config=config
        r=NS(model=NS(config=config),adapter=NS(layers=[NS(self_attn=attention)]),arm='off')
        with patch('inspect.getsource',return_value='eager_attention_forward'):
            self.assertFalse(backend_audit(r)['fallback'])
            config._attn_implementation='sdpa'
            with self.assertRaisesRegex(PrefixError,'expected eager'): backend_audit(r)
        self.assertFalse(validate_attention_output(None,'off',2,2,1,32)['returned'])
        with self.assertRaises(PrefixError): validate_attention_output([NS(shape=(1,2,1,32))],'off',1,2,1,32)
        self.assertTrue(validate_attention_output([NS(shape=(1,2,1,32))],'on',1,2,1,32)['returned'])
        with self.assertRaises(PrefixError): validate_attention_output(None,'on',1,2,1,32)


if __name__=='__main__': unittest.main()
