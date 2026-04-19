# Shortcut-Resistant Multi-view Feature Selection and Semantic Augmentation for Encrypted Traffic Classification

Code for the paper:

**Shortcut-Resistant Multi-view Feature Selection and Semantic Augmentation for Encrypted Traffic Classification**

This repo contains the core pipeline for shortcut-resistant encrypted traffic classification. In short, it covers:

- side-channel feature extraction from traffic traces
- multi-view feature selection with `MUMFS`
- minority-class augmentation with `TAGAN-F`
- risk-aware inference with the layered decision module

The current release uses ISCX as the running example for feature extraction and pipeline execution, while the other datasets follow the same extraction and preprocessing logic.

## Main Components

- `mumfs.py`: MUMFS-based multi-view feature scoring and adaptive feature selection.
- `tagan_f.py`: TAGAN-F, a semantics-aware tabular GAN for minority augmentation.
- `layer_decision.py`: the layered decision module for threshold-based routing and centroid-assisted expert selection.

## Files

- `core_utils.py`: shared utilities for seeding, timing, standardization, and XGBoost training.
- `datasets.py`: ISCX CSV loading, split handling, view construction, and feature export helpers.
- `extract_iscx_ultimate.py`: PCAP-to-CSV feature extraction script for the ISCX-VPN setting.
- `run_iscx_pipeline.py`: end-to-end training and evaluation entry point for the public pipeline.
- `mumfs.py`: multi-view graph construction, sparse hashing scorer, and adaptive selector.
- `tagan_f.py`: conditional WGAN-GP style generator with semantic, correlation, and feature-matching constraints.
- `layer_decision.py`: per-class threshold calibration, uncertainty routing, and centroid-based gating.

## Datasets

- The experiments in the paper are based on public datasets, including [ISCXVPN2016](https://www.unb.ca/cic/datasets/vpn.html), [USTC-TFC2016](https://github.com/davidyslu/USTC-TFC2016), [CSTNET-TLS 1.3](https://github.com/linwhitehat/ET-BERT), and [CipherSpectrum](https://cgi.cse.unsw.edu.au/~cspectrum/).
- The ISCX extractor is provided here as a concrete example. The extraction logic used for the other datasets follows the same overall design principle of shortcut-reduced, side-channel-oriented feature construction.
