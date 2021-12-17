from typing import Optional

from transformers import pipeline


generator = pipeline("text-generation", model="sberbank-ai/rugpt3small_based_on_gpt2")


class LanguageModel:
    """
    TODO-Doc
    """
    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Parameters
        ----------
            model_name : str
                One of possible model names.
        """
        self.model_name = model_name or "sberbank-ai/rugpt3small_based_on_gpt2"
        self.language_model = pipeline("text-generation", model=self.model_name)

    def __call__(
        self,
        prefix: str,
        max_length: int = 100,
        num_return_sequences: str = 1,
        do_sample: bool = True,
        top_k: int = 50
    ) -> str:
        """
        TODO-Doc
        """
        # TODO: сделать top-k и nucl
        generated_struct = generator(
            prefix,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            top_k=top_k,
        )
        # TODO: вот здесь тоже расширить до батчей
        return generated_struct[0]['generated_text']
