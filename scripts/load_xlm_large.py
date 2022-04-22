from transformers import XLMRobertaTokenizer, XLMRobertaModel

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
model = XLMRobertaModel.from_pretrained(
    "xlm-roberta-large", add_pooling_layer=False
)
