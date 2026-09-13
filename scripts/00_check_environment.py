import importlib.metadata
import json
import platform
from _common import ROOT, arguments
from semcache.utils.io import write_json

args, config = arguments('Report runtime and configured model without loading weights')
report = {'python': platform.python_version(), 'packages': {}, 'metric_source': 'measured'}
for package in ('torch', 'transformers', 'peft', 'PyYAML', 'pytest', 'matplotlib'):
    try:
        report['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report['packages'][package] = None
report.update(cuda_available=None, cuda_version=None, devices=[],
              model_id=config['model']['name'], model_dtype=config['model']['dtype'],
              model_revision=config['model']['revision'], resolved_model_revision=None)
if report['packages']['torch']:
    try:
        import torch
        report['cuda_available'] = torch.cuda.is_available()
        report['cuda_version'] = torch.version.cuda
        report['devices'] = [{'name': torch.cuda.get_device_name(i),
                              'total_vram_bytes': torch.cuda.get_device_properties(i).total_memory}
                             for i in range(torch.cuda.device_count())]
    except (ImportError, OSError, RuntimeError) as exc:
        report['torch_runtime_error'] = str(exc)
report['missing_dependencies'] = [p for p in ('torch','transformers') if report['packages'][p] is None]
write_json(args.output or ROOT / 'results/raw/environment.json', report)
print(json.dumps(report, indent=2))
if report['missing_dependencies']:
    print('Missing dependencies: '+', '.join(report['missing_dependencies']))
    print("Install explicitly: python3 -m pip install -e '.[models,test]'")
    raise SystemExit(1)
