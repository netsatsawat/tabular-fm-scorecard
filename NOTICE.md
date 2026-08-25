# Third-party notices

The code in this repository is MIT-licensed. The model weights it downloads are not, and
they do not share one licence. Read this before using any result here for anything other
than research.

## Model weights

| model | code licence | weights licence | commercial use |
|---|---|---|---|
| TabFM 1.0.1 | Apache-2.0 | `tabfm-non-commercial-v1.0` | **No** |
| TabPFN 8.4.0 | Prior Labs License (Apache-2.0 + attribution) | varies by weight version, and TabPFN-3 is non-commercial | **No** for the default |
| TabICL v2.1.1 | BSD-3-Clause | BSD-3-Clause | Yes |
| TabDPT 1.2.0 | Apache-2.0 | Apache-2.0 | Yes |
| Nori 0.18.2 | Apache-2.0 | Apache-2.0 | Yes (regression only) |

### TabFM

The source is Apache-2.0. The pretrained weights are governed separately by the TabFM
Non-Commercial License v1.0, which permits *"testing, evaluation, or research not tied to
commercial gain"* and states that this *"includes internal benchmarking, academic research,
and experimentation on private or public datasets, provided the results are not used in
commercial decision-making, client deliverables, or paid products/services."*

Section 3 also forbids using or distributing *"any Outputs and data produced by the TabFM
Model"* for commercial or production purposes, and forbids redistributing the model or a
derivative at all.

This repository therefore does not vendor any weights, and the numbers it produces for
TabFM are research output. They must not be carried into paid work.

### TabPFN

`pip install tabpfn` defaults to TabPFN-3, whose weights are non-commercial. The permissive
Prior Labs License covers the code and the TabPFN v2 weights, not the current default.
Downloading weights also requires a one-time licence acceptance and a `TABPFN_TOKEN`.

## Dataset

[EV Battery Failure Prediction Dataset (200k)](https://www.kaggle.com/datasets/sarveshchhetri/ev-battery-failure-prediction-dataset-200k),
retrieved via the Kaggle API. Synthetic, by its author's own description. Not redistributed
here. `data/` is gitignored and the download is a documented step.
