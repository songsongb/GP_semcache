"""Cost replay over explicit cache-event traces, without reference-result inputs."""


class EdgeLoRASimulator:
    def __init__(self, cost_model):
        self.cost_model = cost_model

    def run(self, trace, *, model_config, system_config):
        """Replay n/n_reused per query. Does not generate cache events or answers."""
        return [self.cost_model.estimate(tokens=row['query_token_count'],
                    reused_tokens=row['reused_token_count'],model_config=model_config,
                    system_config=system_config) for row in trace]
