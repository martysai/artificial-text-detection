# Data generation description

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
dvc push
```

Do not forget to commit `data.dvc` file in order to pull processed datasets later.


# TODO: refactor later on

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
