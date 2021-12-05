MODELS=('opus-mt-ru-en' 'opus-mt-ru-es' 'opus-mt-ru-fi' 'opus-mt-ru-fr')

cd resources/data
for model in "${MODELS[@]}"; do
  git lfs clone "https://huggingface.co/Helsinki-NLP/${model}"
done
