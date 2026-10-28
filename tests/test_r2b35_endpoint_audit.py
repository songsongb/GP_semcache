import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from semcache.diagnostics.r2b35_endpoint_audit import validate_shapes,EndpointAudit
from semcache.diagnostics.r2b1_prefix_cache import PrefixError


class AuditTests(unittest.TestCase):
    def test_one_head_and_reject_posthoc_and_double_head(self):
        one=dict(input=[1,1,8],output=[1,1,64])
        self.assertEqual(validate_shapes([one],1,32,8,64,True)['head_calls'],1)
        full=dict(input=[1,32,8],output=[1,32,64])
        for heads,calls in [([full],1),([full,one],1),([one],2),([],1)]:
            with self.assertRaises(PrefixError): validate_shapes(heads,calls,32,8,64,True)
        self.assertEqual(validate_shapes([full],1,32,8,64,False)['status'],'LEGACY_ALL_LOGITS')

    def test_missing_or_disabled_active_lora_rejected_and_hooks_cleaned(self):
        class Module:
            def __init__(self): self.handles=[]
            def hook(self,*args,**kwargs):
                h=NS(remove=lambda:self.handles.remove(h)); self.handles.append(h); return h
            register_forward_hook=hook; register_forward_pre_hook=hook
        decoder=Module(); head=Module()
        r=NS(matched_decoder=decoder,matched_head=head,adapter=NS(layers=[1],projection_modules=lambda i:{'q':NS(disable_adapters=True,merged=False,active_adapters=['user_a'])}),a=NS(adapter='user_a'))
        audit=EndpointAudit(r,'MATCHED_FULL')
        with patch.dict('sys.modules',{'torch':NS()}):
            with self.assertRaises(PrefixError):
                with audit.hooks(): pass
        self.assertEqual(decoder.handles,[]); self.assertEqual(head.handles,[])


if __name__=='__main__': unittest.main()
