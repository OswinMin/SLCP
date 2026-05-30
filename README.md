# SLCP

Code for the paper *Stable Localized Conformal Prediction via Transduction*.

This repository contains the core implementation of SLCP, together with the simulation and real-data experiment scripts used in the paper.

## Structure

- `Main/`: main method implementation and supporting modules.
- `SimuAnalysis/`: simulation experiments.
- `RealAnalysis/`: real-data experiments.
- `Dataset/`: datasets used by the experiment scripts.

## Dependencies

This repository does not currently include a `requirements.txt`. Based on the code, the main dependencies include:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `torch`

Some scripts may also require additional dataset-specific support, for example `pyreadstat` for reading `.sav` files via `pandas`.

## Example Usage

Run one simulation setting with:

```bash
python SimuAnalysis/run_shared.py quad 30 500
```

Run real-data experiments with scripts such as:

```bash
python RealAnalysis/Crime.py
python RealAnalysis/Protein.py
python RealAnalysis/Achieve.py
python RealAnalysis/Derma.py
python RealAnalysis/Tissue.py
```

## Notes

- Some image experiments use pretrained checkpoints stored in `RealAnalysis/Para/`.
- Experiment outputs may be written to directories such as `SimResult/`, `Log/`, and `Figure/`.
