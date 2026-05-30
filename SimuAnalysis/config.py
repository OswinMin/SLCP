AVAILABLE_DTYPES = [
    "abs",
    "exp",
    "quad",
    "logabs",
    "relu",
    "softplus",
    "sigmoid",
    "sqrtabs",
    "maxabs",
    "l1",
    "l2",
    "sin",
    "mix",
]

RUN_DTYPES = [
    "abs",
    "exp",
    "quad",
    "logabs",
    "relu",
    "softplus",
    "sigmoid",
    "sqrtabs",
    "maxabs",
    "mix",
]

DISPLAY_DTYPES = ["quad", "softplus", "logabs"]

DTYPE_LABELS = {
    "abs": "DGP 1",
    "exp": "DGP 2",
    "quad": "Quad",
    "logabs": "LogAbs",
    "relu": "ReLU",
    "softplus": "Softplus",
    "sigmoid": "Sigmoid",
    "sqrtabs": "SqrtAbs",
    "maxabs": "MaxAbs",
    "l1": "L1",
    "l2": "L2",
    "sin": "Sin",
    "mix": "Mix",
}

D = 5
R = 0.5
N = 2000
GAMMA_T = 1.0
GAMMA_S = 1.2
ALPHA = 0.1
REPEATS = 50
TESTN = 500
HIDDEN_DIM = [50, 100, 100, 50]
EPOCHES = 200
N_GRID = 50
LBDS = [0.0, 0.0005, 0.00075, 0.001, 0.002, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
TEMPERATURE = 10.0
ALPHA_TOL = 0.02

FIXED_N = 30
FIXED_M = 500
LAMBDA_N = 30
LAMBDA_M = 500

CANDIDATE_N_VALUES = [30, 100, 500]
CANDIDATE_M_VALUES = [30, 100, 500]

DISPLAY_N_VALUES = [30, 100, 500]
DISPLAY_M_VALUES = [30, 100, 500]


def _ordered_unique(values):
    out = []
    seen = set()
    for value in values:
        item = int(value)
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def dtype_label(dtype):
    return DTYPE_LABELS.get(dtype, dtype)


def canonical_pairs():
    pairs = []
    for n in _ordered_unique(CANDIDATE_N_VALUES):
        pairs.append((n, int(FIXED_M)))
    for m in _ordered_unique(CANDIDATE_M_VALUES):
        pairs.append((int(FIXED_N), m))

    out = []
    seen = set()
    for pair in pairs:
        if pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out


def validate_config():
    unknown_run = [dtype for dtype in RUN_DTYPES if dtype not in AVAILABLE_DTYPES]
    unknown_display = [dtype for dtype in DISPLAY_DTYPES if dtype not in AVAILABLE_DTYPES]
    if unknown_run:
        raise ValueError(f"Unknown run dtypes: {unknown_run}")
    if unknown_display:
        raise ValueError(f"Unknown display dtypes: {unknown_display}")

    candidate_n = set(_ordered_unique(CANDIDATE_N_VALUES))
    candidate_m = set(_ordered_unique(CANDIDATE_M_VALUES))
    for n in DISPLAY_N_VALUES:
        if int(n) not in candidate_n:
            raise ValueError(f"display n value {n} is not in candidate_n_values")
    for m in DISPLAY_M_VALUES:
        if int(m) not in candidate_m:
            raise ValueError(f"display m value {m} is not in candidate_m_values")

    if (int(LAMBDA_N), int(LAMBDA_M)) not in canonical_pairs():
        raise ValueError("lambda pair must be contained in canonical_pairs()")


def common_run_kwargs():
    validate_config()
    return {
        "d": int(D),
        "r": float(R),
        "N": int(N),
        "gamma_t": float(GAMMA_T),
        "gamma_s": float(GAMMA_S),
        "alpha": float(ALPHA),
        "repeats": int(REPEATS),
        "testN": int(TESTN),
        "hidden_dim": list(HIDDEN_DIM),
        "epoches": int(EPOCHES),
        "n_grid": int(N_GRID),
        "lbds": list(LBDS),
        "temperature": float(TEMPERATURE),
        "alpha_tol": float(ALPHA_TOL),
        "run_sdcp": True,
        "run_ppi": True,
        "run_noal": True,
        "run_sel": True,
    }
