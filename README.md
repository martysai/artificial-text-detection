[![Build Status][build_status_badge]](build_status_link)
[![codecov](https://codecov.io/gh/MaratSaidov/artificial-text-detection/branch/main/graph/badge.svg?token=HF8IQEADRU)](https://codecov.io/gh/MaratSaidov/artificial-text-detection)
[![DVC](https://img.shields.io/badge/-Data_Version_Control-white.svg?logo=data-version-control&style=social)](https://dvc.org/?utm_campaign=badge)
[![license](https://img.shields.io/github/license/maratsaidov/artificial-text-detection.svg)](https://github.com/maratsaidov/artificial-text-detection/blob/master/LICENSE)

# Artificial Text Detection
Solutions for true/generated text distinction.

### Contents

Project description is put into:

- [Framework Description Markdown](https://github.com/MaratSaidov/artificial-text-detection/blob/main/detection/README.md);
- [Data Description Markdown](https://github.com/MaratSaidov/artificial-text-detection/blob/main/detection/data/README.md).

### Installation steps:

```bash
make poetry-download
make install
make lint
```

### Data Version Control usage:

```bash
poetry add "dvc[gdrive]"
```

Check out `dvc.yaml` and `parameters.yaml` for a better pipelines understanding.


### Simple run:

```bash
python detection/pipeline.py
```

[build_status_badge]: https://github.com/MaratSaidov/artificial-text-detection/actions/workflows/build.yml/badge.svg
[build_status_link]: https://github.com/MaratSaidov/artificial-text-detection/actions/workflows/build.yml
