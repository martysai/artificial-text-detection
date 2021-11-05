[![Build Status][build_status_badge]](build_status_link)
[![codecov](https://codecov.io/gh/MaratSaidov/artificial-text-detection/branch/main/graph/badge.svg?token=HF8IQEADRU)](https://codecov.io/gh/MaratSaidov/text-detection)
[![DVC](https://img.shields.io/badge/-Data_Version_Control-white.svg?logo=data-version-control&style=social)](https://dvc.org/?utm_campaign=badge)

# Artificial Text Detection
Solutions for true/generated text distinction.

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
