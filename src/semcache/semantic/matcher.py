from abc import ABC, abstractmethod


class SubsequenceMatcher(ABC):
    @abstractmethod
    def key(self, cluster_id, subsequence):
        """Return a cache index key, not evidence of numerical reusability."""


class ExactTokenMatcher(SubsequenceMatcher):
    match_rule = "exact_token_ids_within_cluster"

    def key(self, cluster_id, subsequence):
        return cluster_id, tuple(subsequence.token_ids)


def make_matcher(rule):
    if rule != ExactTokenMatcher.match_rule:
        raise ValueError(f"Unsupported matching policy: {rule}")
    return ExactTokenMatcher()
