# RedNet: inference code and data on redesigning selective protein binders.

RedNet is a sequence redesign model for one-sided protein interface design, focused on improving binding affinity and target specificity.

## Installation

Requires Python ≥ 3.10 and a CUDA-enabled GPU for inference.

```bash
git clone https://github.com/zw2x/rednet_public.git
cd rednet_public
pip install -e .
```

## Usage

Inference entry points are exposed under `cli/`:

```bash
# Selective binder redesign
python cli/infer_pipeline.py run_sel --config configs/sampling/<config>.yaml

# Heterodimer self-consistency test
python cli/infer_pipeline.py run_hdimer --config configs/sampling/<config>.yaml

# Folding / scoring pipeline
python cli/fold_pipeline.py --help

# Analysis and metric aggregation
python cli/anal_pipeline.py --help
```

## Repository layout

```
src/rednet/         Model, data, sampling, and Lightning task code
cli/                Command-line entry points (inference, folding, analysis)
configs/            Hydra/OmegaConf configs (model, dataset, sampling, eval)
docs/               Filter thresholds and benchmark CSVs
assets/             Figures and supplementary data tables
```

## Data

Benchmark CSVs are in `docs/` (SKEMPI subset, heterodimer set, test PDB list). Model weights and selectivity benchmarks are found [here](https://zenodo.org/records/20113403).

## License

Apache License 2.0. See [LICENSE](LICENSE).
