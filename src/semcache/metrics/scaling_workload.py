"""Deterministic natural-text fixtures for M8 prompt-length scaling."""
from collections import defaultdict

HOTEL_TEXT = (
    "A traveler is planning a careful visit to Cambridge and needs a quiet hotel near the railway station. "
    "The room should be comfortable, reasonably priced, and convenient for an early morning departure. "
    "Please compare practical lodging choices and keep the booking details clear for the guest."
)

UNRELATED_TEXT = (
    "A community gardener records the seasonal weather while caring for vegetables, fruit trees, and flowers. "
    "The notes describe rainfall, afternoon sunlight, soil moisture, and the best time to protect young plants. "
    "Please organize the observations into a useful plan for the coming week."
)


def _token_ids(tokenizer, text):
    return list(tokenizer(text)["input_ids"])


def construct_natural_prompt(tokenizer, target_tokens, *, topic="hotel"):
    """Choose the closest tokenized prefix of a repeated natural paragraph."""
    if type(target_tokens) is not int or target_tokens < 4:
        raise ValueError("target_tokens must be an integer of at least four")
    paragraph = HOTEL_TEXT if topic == "hotel" else UNRELATED_TEXT if topic == "unrelated" else None
    if paragraph is None:
        raise ValueError("Unknown prompt topic")
    words = paragraph.split()
    best = None
    # Tokenizer-aware construction is outside every measured request.
    for count in range(1, target_tokens * 3 + len(words) + 1):
        text = " ".join(words[i % len(words)] for i in range(count))
        ids = _token_ids(tokenizer, text)
        candidate = (abs(len(ids) - target_tokens), len(ids), text, ids)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        if len(ids) >= target_tokens:
            break
    _, actual, text, ids = best
    return dict(text=text, requested_prompt_tokens=target_tokens,
                actual_prompt_tokens=actual, token_ids=ids, topic=topic)


def controlled_length_trace(tokenizer, target_tokens):
    exact = construct_natural_prompt(tokenizer, target_tokens, topic="hotel")
    unrelated = construct_natural_prompt(tokenizer, target_tokens, topic="unrelated")
    exact["subsequence_audit"] = subsequence_occurrence_report(exact["token_ids"])
    unrelated["subsequence_audit"] = subsequence_occurrence_report(unrelated["token_ids"])
    trace = [
        dict(condition="cold_miss", user_id="user_a", **exact),
        dict(condition="same_user_exact", user_id="user_a", **exact),
        dict(condition="cross_user_exact", user_id="user_b", **exact),
        dict(condition="unrelated", user_id="user_a", **unrelated),
    ]
    if not (trace[0]["token_ids"] == trace[1]["token_ids"] == trace[2]["token_ids"]):
        raise AssertionError("Exact-repeat conditions must have identical token IDs")
    return trace


def parse_prompt_lengths(value):
    if value is None or not value.strip():
        return []
    try:
        lengths = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("Prompt lengths must be comma-separated integers") from exc
    if not lengths or len(set(lengths)) != len(lengths) or any(x not in (32, 64, 128, 256, 512) for x in lengths):
        raise ValueError("Prompt lengths must be unique values from 32,64,128,256,512")
    return lengths


def subsequence_occurrence_report(token_ids, window_size=3):
    positions = defaultdict(list)
    for start in range(max(0, len(token_ids) - window_size + 1)):
        positions[tuple(token_ids[start:start + window_size])].append(start)
    duplicated = [dict(token_ids=list(key), positions=starts, occurrences=len(starts))
                  for key, starts in positions.items() if len(starts) > 1]
    duplicated.sort(key=lambda item: (item["positions"][0], item["token_ids"]))
    return dict(total_windows=sum(len(x) for x in positions.values()),
        unique_windows=len(positions), duplicated_key_count=len(duplicated),
        max_occurrences=max((len(x) for x in positions.values()), default=0),
        duplicated_windows=duplicated)
