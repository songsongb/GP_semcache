from _common import ROOT, arguments
from semcache.models.loader import load_model
from semcache.models.lora_fixtures import create_controlled_users
from semcache.models.model_adapter import OPTModelAdapter
from semcache.utils.seed import seed_everything
from semcache.semantic.encoder import ControlledEncoder, HuggingFaceTextEncoder, TinyBERTSemanticEncoder
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.cache.global_cache import GlobalCache
from semcache.cache.admission import AdmissionPolicy
from semcache.cache.eviction import EvictionPolicy
from semcache.semcache_engine import SemCacheEngine
from semcache.evaluation.mixed_control import CONTROL_TEXT

TRACE = [
    ('health_source', 'user_a', CONTROL_TEXT, [0., 0.]),
    ('health_target', 'user_b', 'Today I need clear health advice about sleep and exercise please.', [0., 0.]),
    ('health_repeat', 'user_b', CONTROL_TEXT, [0., 0.]),
    ('weather_source', 'user_a', 'I need clear weather advice about rain and wind today please.', [10., 10.]),
    ('weather_target', 'user_b', 'Today I need clear weather advice about rain and wind please.', [10., 10.]),
]


def setup():
    def options(parser):
        parser.set_defaults(config=ROOT/'configs/development.yaml')
        parser.add_argument('--compare-baseline', action='store_true')
        parser.add_argument('--logical-capacity-bytes', type=int, help='Development-only forced eviction override')
    args, config = arguments('Controlled integrated SemCache prefill; synthetic adapters, no safety claim', options)
    if args.layers is not None:
        raise ValueError('M5 validates all layers; do not supply --layers')
    seed_everything(config['seed'])
    model, tokenizer, metadata = load_model(config['model'])
    model, fixtures = create_controlled_users(model, config.get('lora'))
    return args, config, model, tokenizer, metadata, fixtures, OPTModelAdapter(model)


def engine_for(config, model, tokenizer, metadata, adapter, capacity=None):
    c = config['semcache']
    if (config['match_rule'] != 'exact_token_ids_within_cluster'
            or config['cache']['normalization'] != 'current_pool_plus_candidate'
            or config['cache']['zero_max'] != 0 or config['cache']['age_unit'] != 'query'
            or config['clustering']['update_rule'] != 'incremental_mean_after_every_assignment'
            or config['clustering']['update_mode'] not in ('immediate_eq9', 'buffered')
            or c['overlap_policy'] != 'earliest_start_then_impact_then_key'
            or c['pbr_trigger'] not in ('manual', 'interval')):
        raise ValueError('Unsupported M5 reproduction policy')
    if c['encoder_kind'] == 'fixture':
        encoder = ControlledEncoder({q: v for _, _, q, v in TRACE})
        centroids = c['fixture_centroids']
    elif c['encoder_kind'] == 'huggingface_text':
        encoder = HuggingFaceTextEncoder(**config['semantic_encoder'])
        centroids = c['initial_centroids']  # Must match explicitly configured encoder dimensions.
    elif c['encoder_kind'] == 'tinybert':
        encoder = TinyBERTSemanticEncoder(**config['semantic_encoder'])
        centroids = c['initial_centroids']  # Explicit warmup embeddings are also supported by script 29.
    else:
        raise ValueError('Unsupported semantic encoder kind')
    clusterer = IntentClusterer(len(centroids), config['cluster_update_interval_queries'],
                                update_mode=config['clustering']['update_mode'])
    clusterer.initialize(centroids, counts=[0]*len(centroids))
    cache = GlobalCache(capacity if capacity is not None else int(config['logical_cache_capacity_gb']*10**9),
                        AdmissionPolicy(**config['admission']), EvictionPolicy(**config['eviction']))
    return SemCacheEngine(model, tokenizer, adapter, encoder,
        clusterer, cache, window_size=config['subsequence_window'], storage_device=c['physical_storage_device'],
        rho=config['semantic_impact']['rho'], history_lambda=config['semantic_impact']['history_lambda'],
        frequency_window=config['cache']['admission_frequency_window_queries'], metadata=metadata, seed=config['seed'],
        impact_reducer=config['semantic_impact'].get('reducer', 'paper_row_l2_sum'),
        pbr_interval_queries=(config['semantic_impact']['pbr_interval_queries']
                              if c['pbr_trigger'] == 'interval' else None))
