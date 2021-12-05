[![Build Status][build_status_badge]](build_status_link)
[![codecov](https://codecov.io/gh/MaratSaidov/artificial-text-detection/branch/main/graph/badge.svg?token=HF8IQEADRU)](https://codecov.io/gh/MaratSaidov/artificial-text-detection)
[![DVC](https://img.shields.io/badge/-Data_Version_Control-white.svg?logo=data-version-control&style=social)](https://dvc.org/?utm_campaign=badge)
[![license](https://img.shields.io/github/license/maratsaidov/artificial-text-detection.svg)](https://github.com/maratsaidov/artificial-text-detection/blob/master/LICENSE)

# Artificial Text Detection
Solutions for true/generated text distinction.

### Contents

Project description is put into:

- [Framework Description Markdown](https://github.com/MaratSaidov/artificial-text-detection/blob/main/detection/README.md)
- [Data Description Markdown](https://github.com/MaratSaidov/artificial-text-detection/blob/main/detection/data/README.md)
- [Models Description Markdown](https://github.com/MaratSaidov/artificial-text-detection/blob/main/detection/data/README.md)

### Installation steps:

We use [`poetry`](https://python-poetry.org/) as an enhanced dependency resolver.

```bash
make poetry-download
poetry install --no-dev
```

### Datasets for artificial text detection

To create datasets for the further classification, it is necessary to collect them.
There are 2 available ways for it:

- Via [Data Version Control](https://dvc.org/).
Get in touch with [`@msaidov`](https://t.me/msaidov) in order to have the access to the private Google Drive;
- Via datasets generation. One dataset with a size of 20,000 samples was process with MT model on V100 GPU for 30 mins;

### Data Version Control usage:

```bash
poetry add "dvc[gdrive]"
```

Then, run `dvc pull`. It will download preprocessed translation datasets
from the Google Drive.

### Datasets generation

To generate translations before artificial text detection pipeline,
install the `detection` module from the cloned repo or PyPi (TODO):
```bash
pip install -e .
```
Then, run generate script:
```bash
python detection/data/generate.py --dataset_name='tatoeba' --size=20000 --device='cuda:0'
```

### Simple run:

To run the artificial text detection classifier, execute the pipeline:

```bash
python detection/pipeline.py
```

[build_status_badge]: https://github.com/MaratSaidov/artificial-text-detection/actions/workflows/build.yml/badge.svg
[build_status_link]: https://github.com/MaratSaidov/artificial-text-detection/actions/workflows/build.yml
