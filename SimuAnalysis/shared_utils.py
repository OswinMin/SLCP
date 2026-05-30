import pickle
from pathlib import Path

import config


SIM_DIR = Path(__file__).resolve().parent
REPO_DIR = SIM_DIR.parent


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def shared_result_dir():
    return REPO_DIR / "SimResult" / "SimuAnalysis" / "shared"


def figure_dir():
    return REPO_DIR / "Figure" / "SimuAnalysis"


def command_dir():
    return SIM_DIR / "command"


def output_dir():
    return SIM_DIR / "output"


def _tag(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def make_result_stem(dtype, n, m, gamma_s=None, r=None):
    gamma_s = config.GAMMA_S if gamma_s is None else gamma_s
    r = config.R if r is None else r
    return f"Shared_{dtype}_n{int(n)}_m{int(m)}_g{_tag(float(gamma_s))}_r{_tag(float(r))}"


def result_path(dtype, n, m, gamma_s=None, r=None):
    return shared_result_dir() / f"{make_result_stem(dtype, n, m, gamma_s=gamma_s, r=r)}.pkl"


def save_result(dtype, n, m, res, gamma_s=None, r=None):
    path = result_path(dtype, n, m, gamma_s=gamma_s, r=r)
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(res, f)
    return path


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_result(dtype, n, m, gamma_s=None, r=None):
    path = result_path(dtype, n, m, gamma_s=gamma_s, r=r)
    if not path.exists():
        raise FileNotFoundError(f"Missing shared result: {path}")
    return load_pickle(path)


def iter_results():
    folder = shared_result_dir()
    if not folder.exists():
        return
    for path in sorted(folder.glob("*.pkl")):
        yield path, load_pickle(path)
