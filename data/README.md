# HMC v1.1 local data

Use existing downloaded files. This repository contains official file names,
checksums, license text and aggregate metadata only, with no PSG or scoring files.

Source: https://physionet.org/content/hmc-sleep-staging/1.1/

Check native EDF units and montage through the supplied reader. The local study
uses 151 recordings; no verified patient linkage across nights was available.
Stage annotation protocol is AASM v2.4. Two scoring formats represent the same
annotations, not two independent raters.

`scripts/download.py` prints acquisition instructions; it does not download 15 GB.
Keep raw EDFs and scoring files in a local directory excluded from Git.
