from abc import ABC, abstractmethod


class CostModel(ABC):
    @abstractmethod
    def estimate(self, *, tokens, reused_tokens, model_config, system_config):
        """Future Eq. 18-20 result; each returned metric must say simulated."""
