# Data generation

Here is a list of supported datasets for this research:

* [Tatoeba](https://huggingface.co/datasets/tatoeba)
* [WikiMatrix](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix)

# Process datasets into DVC:

```bash
python factory.py
dvc add ../../resources/data
```

Do not forget to commit `data.dvc` file in order to pull processed datasets later.
