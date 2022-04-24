from transformers import XLMRobertaModel, XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
model = XLMRobertaModel.from_pretrained(
    "xlm-roberta-large", add_pooling_layer=False
)
# $HOME/atd-models/xlm-roberta-large
