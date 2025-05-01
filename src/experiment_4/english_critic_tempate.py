from typing import ClassVar

from .base_critic_template import BaseCriticTemplate


class EnglishCriticTemplate(BaseCriticTemplate):
    """English chat template for generating feedback on prompt that is being constructed to generate some additional examples for a given intent class."""

    _GENERATE_INSTRUCTION: ClassVar[str] = \
        "You are a professional prompt engineer. You have to help your colleague with his prompt engineering task. " \
        "The task is to construct a prompt that will make an LLM generate new samples for the dataset. " \
        "Dataset consists of utterances (texts) and labels that correspond to them (intents). " \
        "Separate prompt is constructed for each intent to enrich the original data with LLM-generated utterances of particular intent. " \
        "You will be given the current prompt for intent {intent_name}. " \
        "You have to criticize this prompt and suggest ideas for improving it. " \
        "Prompt: {prompt}"
