# Data generation

Some additional information about datasets.

### Data naming rules:

We use the following patterns for datasets naming:

```
factory.py output:
    {dataset_name}.bin
generate.py output:
    {dataset_name}.csv
    {dataset_name}.[train/eval].gen.pth
    {dataset_name}.{lang1}-{lang2}.[train/eval].gen.pth
```

- `factory.py:{dataset_name}.bin` output is a `zlib`-compressed output of `datasets.load_dataset`.
- `generate.py:{dataset_name}.csv` output is a processed source dataset via `EasyNMT` model.
This dataset consists of 3 columns: `sources`, `targets`, `translations`.
- `generate.py:{dataset_name}.[train/eval].gen.pth` is currently a redundant file.
- `{dataset_name}.{lang1}-{lang2}.[train/eval].gen.pth` is currently a redundant file.


### Process datasets into DVC:

```bash
python factory.py
dvc add ../../resources/data
```

Do not forget to commit `data.dvc` file in order to pull processed datasets later.
