from _common import ROOT, arguments
from semcache.evaluation.system_validation import validate_frontend
from semcache.utils.io import write_csv

if __name__ == '__main__':
    args, config = arguments('Fixture semantic frontend correctness (no download)')
    rows = validate_frontend()
    write_csv(args.output or ROOT/'results/raw/semantic_frontend_validation.csv', rows)
    print(rows[0])
