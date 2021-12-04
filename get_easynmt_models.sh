BOOKS=('opus-mt-ru-en' 'opus-mt-ru-es' 'opus-mt-ru-fi' 'opus-mt-ru-fr')

cd resources/data
for book in "${BOOKS[@]}"; do
  git lfs clone "https://huggingface.co/Helsinki-NLP/${book}"
done
