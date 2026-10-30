"""Optional local tiny-OPT API check. No downloads, LoRA/GPU qualification."""
import importlib.util
import unittest

AVAILABLE=all(importlib.util.find_spec(n) for n in ('torch','transformers','numpy'))


@unittest.skipUnless(AVAILABLE,'torch/transformers/numpy absent; actual installed OPT/cache API unverified locally')
class OPTContract(unittest.TestCase):
    def test_eager_attention_return_flag_and_raw_kv_layout(self):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        from semcache.edgelora.r4b2_kv import verify_raw_kv,attention_kv,construct_past
        from semcache.edgelora.r4a_generation import cache_pairs,past_summary
        torch.manual_seed(42)
        config=OPTConfig(vocab_size=17,hidden_size=16,word_embed_proj_dim=16,ffn_dim=32,num_hidden_layers=2,num_attention_heads=4,
                         max_position_embeddings=512,dropout=0.,attention_dropout=0.)
        config._attn_implementation='eager';model=OPTForCausalLM(config).eval();ids=torch.tensor([[1+i%15 for i in range(256)]])
        raw={};handles=[]
        for l,layer in enumerate(model.model.decoder.layers):
            for r in 'qkv':
                def observe(module,args,out,l=l,r=r):raw.setdefault(l,{})[r]=out.detach().clone()
                handles.append(getattr(layer.self_attn,r+'_proj').register_forward_hook(observe))
        with torch.no_grad():source=model(ids[:,:128],attention_mask=torch.ones(1,128,dtype=torch.long),use_cache=True,output_attentions=False)
        for h in handles:h.remove()
        blocks={l:tuple(b[r] for r in 'qkv') for l,b in raw.items()}
        identity=dict(layers=2,hidden_size=16,hidden_dtype='float32')
        proof=verify_raw_kv(blocks,source.past_key_values,4,identity,'cpu')
        past=construct_past(attention_kv(blocks,4,16),type(source.past_key_values),proof['cache_api'])
        with torch.no_grad():
            suffix=model(ids[:,128:],attention_mask=torch.ones(1,256,dtype=torch.long),past_key_values=past,use_cache=True,output_attentions=False)
            full=model(ids,attention_mask=torch.ones_like(ids),use_cache=True,output_attentions=False)
            with_weights=model(ids,attention_mask=torch.ones_like(ids),use_cache=True,output_attentions=True)
        self.assertTrue(torch.equal(full.logits,with_weights.logits))
        self.assertEqual(past_summary(suffix.past_key_values,256,2,4,16,'float32','cpu')['sequence_length'],256)
        self.assertEqual(list(suffix.logits.shape),[1,128,17])
