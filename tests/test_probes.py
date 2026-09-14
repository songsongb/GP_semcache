import pytest
from semcache.semantic.probes import select_pair, construct_probes


def test_full_sequence_controls():
    a = [0,1,2,3,4,5]
    for case,b in [('A',[0,1,2,3,4,6]), ('B',[0,9,2,3,4,6]),
                   ('C',[0,9,8,2,3,4,6]), ('D',[0,1,9,8,7,6])]:
        wa,wb = select_pair(a,b,3,case,special_ids=[0])
        assert wa.token_ids == tuple(a[wa.start:wa.end])
        assert wb.token_ids == tuple(b[wb.start:wb.end])
        assert (wa.token_ids == wb.token_ids) == (case != 'D')
        assert (wa.start == wb.start) == (case != 'C')
        assert (a[:wa.start] == b[:wb.start]) == (case in 'AD')
    with pytest.raises(ValueError):
        select_pair(a,a,3,'B')


class FixtureTokenizer:
    all_special_ids = [0]
    def __call__(self, text):
        # Deterministic complete-prompt character tokenizer, no model claim.
        return {'input_ids': [0]+[ord(c)+1 for c in text]}


def test_deterministic_probe_construction():
    tokenizer = FixtureTokenizer()
    first = construct_probes(tokenizer, 3)
    assert first == construct_probes(tokenizer, 3)
    assert [p['probe_case'] for p in first] == list('ABCD')


def test_negative_control_metadata():
    case = construct_probes(FixtureTokenizer(), 3)[3]
    assert case['probe_case'] == 'D'
    assert case['valid_reuse_candidate'] is False
    assert 'Negative/stress' in case['description']
