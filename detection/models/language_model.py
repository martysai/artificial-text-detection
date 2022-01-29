from typing import Any, Optional, List

from transformers import pipeline


class LanguageModel:
    """
    TODO-Doc
    """

    def __init__(self, model_name: Optional[str] = None, generator: Any = None) -> None:
        """
        Parameters
        ----------
        model_name : str
            One of possible model names.
        generator : Any
            Language model a.k.a generator.
        """
        self.model_name = model_name or "sberbank-ai/rugpt3small_based_on_gpt2"
        self.language_model = pipeline("text-generation", model=self.model_name)
        self.generator = generator or pipeline("text-generation", model="sberbank-ai/rugpt3small_based_on_gpt2")

    def __call__(
        self,
        prefixes: List[str],
        max_length: int = 100,
        num_return_sequences: str = 1,
        do_sample: bool = True,
        top_k: int = 50,
    ) -> List[str]:
        """
        TODO-Doc
        """
        # TODO: сделать top-k и nucl
        generated_struct = self.generator(
            prefixes,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            top_k=top_k,
        )
        return [generated_struct[i][0]["generated_text"] for i in range(len(prefixes))]
