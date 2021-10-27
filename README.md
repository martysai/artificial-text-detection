[![Build Status][build_status_badge]](build_status_link)
[![codecov](https://codecov.io/gh/MaratSaidov/text-detection/branch/main/graph/badge.svg?token=HF8IQEADRU)](https://codecov.io/gh/MaratSaidov/text-detection)


# text-detection
Solutions for true/generated text distinction.

### Roadmap:

- [X] Add tests data
- [X] Set up CI/CD testing
- [X] Add tests for other functions
- [X] Run on remote server
- [X] Add run on the specific size
- [X] Debug the load model error in vast.ai service
- [X] Check why kaggle run doesn't draw plots for wandb.ai
- [X] Separate preprocess and run: finish with dataset saving in generate.py and loading in run.py
- [X] Add `zlib.compress` to `pickle.dump`
- [ ] Metrics understanding
- [ ] Model improvement
- [X] Test passing in github actions

[build_status_badge]: https://github.com/maratsaidov/text-detection/actions/workflows/python-package.yml/badge.svg
[build_status_link]: https://github.com/maratsaidov/text-detection/actions/workflows/python-package.yml
