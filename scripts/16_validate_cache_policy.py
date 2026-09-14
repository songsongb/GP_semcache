from _common import ROOT, arguments
from semcache.evaluation.system_validation import validate_policy
from semcache.utils.io import write_csv

if __name__ == '__main__':
    args, config = arguments('Numerical cache policy correctness; tiny development capacity')
    rows = validate_policy()
    write_csv(args.output or ROOT/'results/raw/cache_policy_validation.csv', rows)
    print(rows[0])
