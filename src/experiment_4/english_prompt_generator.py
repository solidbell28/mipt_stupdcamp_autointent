from typing import ClassVar

from .base_prompt_generator import BasePromptGeneratorTemplate


class EnglishPromptGeneratorTemplate(BasePromptGeneratorTemplate):
    """English chat template for automatic prompt generation for the task of generation of additional examples for the given intent."""

    _GENERATE_INITIAL_INSTRUCTION: ClassVar[str] = \
        "You are a professional prompt engineer. " \
        "You have to construct a prompt that will make an LLM generate new samples for the dataset. " \
        "Dataset consists of utterances (texts) and labels that correspond to them (intents). " \
        "Separate prompt is constructed for each intent to enrich the original data with LLM-generated utterances of particular intent. " \
        "You have to construct a prompt for intent {intent_name}. LLM should return exactly {n_examples} new enumerated utterances. " \
        "Generated utterances should have the same style as examples listed below, but they should differ from them so that " \
        "generated data would bring some variety to the dataset. " \
        "You can use in your prompt the following utterances examples from the original dataset: \n " \
        "{initial_examples} \n " \
        "Return only constructed prompt."

    _GENERATE_INSTRUCTION: ClassVar[str] = \
        "Your colleague reviewed your prompt and provided some remarks and suggestions for improvement. " \
        "Improve your prompt using this information. \n " \
        "Review: {critic_text} \n " \
        "Do not forget to use the following examples in your prompt: \n " \
        "{initial_examples} \n " \
        "Return only constructed prompt."
