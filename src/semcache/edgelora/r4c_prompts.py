"""Model-independent preparation boundary; tokenizer adapters never build model KV."""
from .communication import digest, digest_bytes, require

TEMPLATES = {
    'qasper': ('Answer briefly using the scientific article. Say unanswerable if unsupported; answer yes/no when appropriate.\nArticle:\n',
               '\nQuestion: {question}\nAnswer:'),
    '2wikimqa': ('Answer the question using the passages. Give only the answer.\nPassages:\n', '\nQuestion: {question}\nAnswer:'),
    'multi_news': ('Summarize the following news passages.\nNews:\n', '\n{question}\nWrite a concise summary.\nSummary:')}
TEMPLATE_ID = digest(dict(version='R4C_DERIVED_V1', templates=TEMPLATES))


class TokenizerAdapter:
    def __init__(self, tokenizer, identity, supported_context_length):
        self.tokenizer = tokenizer; self.identity = identity; self.supported_context_length = supported_context_length
        require(type(supported_context_length) is int and supported_context_length > 0, 'actual model positional limit required')

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


class FixtureTokenizer:
    """Explicit test-only reversible character tokenizer. Never OPT evidence."""
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def decode(self, ids, **kwargs): return ''.join(chr(t) for t in ids)


def prepare(row, tokenizer, *, input_budget=256, max_new_tokens=32, enable_512=False):
    require(1 <= input_budget <= (512 if enable_512 else 256), '512-token extension requires explicit opt-in; larger inputs unsupported')
    require(1 <= max_new_tokens <= 128, 'bounded generation budget required')
    require(input_budget + max_new_tokens <= tokenizer.supported_context_length, 'POSITIONAL_LIMIT: input + generation exceeds actual loaded model config')
    original = row['original']; before, after = TEMPLATES[row['dataset']]
    after = after.format(question=original['input'])
    require(len(tokenizer.encode(before + after)) < input_budget, 'INELIGIBLE_PROMPT: question/instruction exceed budget; not truncated')
    context_ids = tokenizer.encode(original['context']); full = before + original['context'] + after
    original_ids = tokenizer.encode(full); keep = len(context_ids)
    # Deterministic head/tail context reduction. Re-tokenize complete final text
    # because BPE boundaries can change at the context/template junctions.
    while True:
        head = (keep + 1) // 2; tail = keep // 2
        spans = [[0, head]] + ([[len(context_ids)-tail, len(context_ids)]] if tail else [])
        retained = original['context'] if keep == len(context_ids) else tokenizer.decode(context_ids[:head]) + ('\n[...context omitted...]\n' if keep < len(context_ids) else '') + (tokenizer.decode(context_ids[-tail:]) if tail else '')
        text = before + retained + after; ids = tokenizer.encode(text)
        if len(ids) <= input_budget: break
        require(keep > 0, 'INELIGIBLE_PROMPT: retained question/instruction and omission marker exceed budget')
        keep -= max(1, min(keep, len(ids)-input_budget))
    require(ids and len(ids)+max_new_tokens <= tokenizer.supported_context_length, 'invalid prepared positional length')
    result = dict(row, sample_id=row['dataset'] + ':' + original['_id'], evaluation_label='LONGBENCH_DERIVED_TRUNCATED_EVALUATION',
        model_tokenizer=tokenizer.identity, model_supported_context_length=tokenizer.supported_context_length,
        prompt_template_id=TEMPLATE_ID, truncation_policy='context-token-head-tail-v1; whole-prompt-retokenization; question/instruction unchanged',
        tokenization=dict(add_special_tokens=False,padding=False,truncation='explicit context-only head/tail followed by full prompt tokenization'),
        original_context_tokens=len(context_ids), retained_context_tokens=keep, removed_context_tokens=len(context_ids)-keep,
        retained_context=retained, retained_context_token_spans=spans, original_formatted_prompt_tokens=len(original_ids),
        question_survived_unchanged=True, instruction_survived_unchanged=True, input_budget=input_budget,
        prepared_input_length=len(ids), max_new_tokens=max_new_tokens, token_ids=ids, attention_mask=[1]*len(ids),
        positions=list(range(len(ids))), prompt_text=text, final_prompt_text_sha256=digest_bytes(text.encode()),
        prompt_sha256=digest(dict(tokens=ids, mask=[1]*len(ids))))
    result['prepared_record_sha256'] = digest(result)
    return result


def validate_prepared(row):
    copy = dict(row); seal = copy.pop('prepared_record_sha256')
    require(digest(copy) == seal, 'prepared record SHA-256 differs')
    ids = row['token_ids']; require(ids and all(type(t) is int and t >= 0 for t in ids), 'invalid actual token IDs')
    require(row['prepared_input_length'] == len(ids) and row['attention_mask'] == [1]*len(ids) and row['positions'] == list(range(len(ids))), 'unpadded prompt identity differs')
    require(row['prompt_sha256'] == digest(dict(tokens=ids, mask=row['attention_mask'])), 'prompt token/mask digest mismatch')
    require(len(ids)+row['max_new_tokens'] <= row['model_supported_context_length'], 'prepared positional limit differs')
    return row


def prefix_of(target, length):
    validate_prepared(target); require(type(length) is int and 0 < length < len(target['token_ids']), 'unsupported exact prefix boundary')
    ids = target['token_ids'][:length]  # Never decode/re-encode prefix.
    return dict(sample_id=target['sample_id']+':prefix', token_ids=ids, attention_mask=[1]*length, positions=list(range(length)),
                prompt_sha256=digest(dict(tokens=ids, mask=[1]*length)), target_prepared_sha256=target['prepared_record_sha256'],
                prefix_origin='ACTUAL_TARGET_INITIAL_TOKEN_IDS', target_sample_id=target['sample_id'])
