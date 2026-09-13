import importlib.util


def require_models():
    missing = [name for name in ('torch', 'transformers') if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit('Missing dependencies: '+', '.join(missing)+". Install: python3 -m pip install -e '.[models,test]'")
