import sys

import config
from core import run_experiment
from shared_utils import make_result_stem, save_result


def main(argv):
    if len(argv) != 4:
        raise SystemExit("Usage: python SimuAnalysis/run_shared.py <dtype> <n> <m>")

    dtype = argv[1]
    n = int(argv[2])
    m = int(argv[3])

    kwargs = config.common_run_kwargs()
    res = run_experiment(dtype=dtype, n=n, m=m, **kwargs)
    res["meta"]["result_name"] = make_result_stem(dtype, n, m)
    res["meta"]["result_group"] = "shared"

    out_path = save_result(dtype, n, m, res)
    print(f"[Done] saved: {out_path}")


if __name__ == "__main__":
    main(sys.argv)
