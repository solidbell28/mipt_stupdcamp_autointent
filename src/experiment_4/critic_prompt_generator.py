from typing import List

from .base_prompt_generator import BasePromptGeneratorTemplate
from .base_critic_template import BaseCriticTemplate

from autointent.generation import Generator
from autointent import Dataset
from autointent.generation.chat_templates._evolution_templates_schemas import Message, Role
from autointent.schemas import Intent, Sample
from autointent.custom_types import Split

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets


class PromptCriticGenerator:
    """Generator of new utterances based on exiting ones using the prompt generated using LLM-based critic workflow."""

    def __init__(
        self, generator: Generator, prompt_maker: BasePromptGeneratorTemplate,
        critic_prompt_maker: BaseCriticTemplate, async_mode: bool = False
    ) -> None:
        """
        Initialize the PromptCriticGenerator.

        Args:
            generator: Wrapper to access the LLM API.
            prompt_maker: Generator of prompts for the prompt-engineer LLM.
            critic_prompt_maker: Generator of prompts for the critic LLM.
            async_mode: Whether to use asynchronous requests to LLM API or not (added only for interface consistency).
        """
        self.generator = generator
        self.prompt_maker = prompt_maker
        self.critic_prompt_maker = critic_prompt_maker
        self.async_mode = async_mode

    def __call__(
        self, intent_data: Intent, n_generations: int, n_hops: int
    ) -> List[str]:
        """
        Call the generator to generate new utterances.

        Args:
            intent_data: Intent data for which to generate utterances.
            n_generations: Number of utterances to generate.
            n_hops: Number of iterations of generation-criticism mechanism.

        Returns:
            List of generated utterances.
        """
        messages = self.prompt_maker(intent_data, None, n_generations)
        response_text = self.generator.get_chat_completion(messages)
        messages.append(Message(role=Role.ASSISTANT, content=response_text))

        for hop in range(n_hops):
            critic_messages = self.critic_prompt_maker(
                intent_data, response_text
            )
            critic_response_text = self.generator.get_chat_completion(
                critic_messages
            )
            cur_messages = self.prompt_maker(
                intent_data, critic_response_text, n_generations
            )
            messages.extend(cur_messages)
            response_text = self.generator.get_chat_completion(messages)
            messages.append(
                Message(role=Role.ASSISTANT, content=response_text)
            )

        final_message = [Message(role=Role.USER, content=response_text)]
        final_response_text = self.generator.get_chat_completion(final_message)

        return _extract_utterances(final_response_text)

    def augment(
        self, dataset: Dataset, split_name: str = Split.TRAIN,
        n_generations: int = 5, n_hops: int = 3, update_split: bool = True
    ) -> List[Sample]:
        """
        Apply augmentations to the dataset.

        Args:
            dataset: Dataset for which to apply augmentations.
            split_name: Dataset split for which to apply augmentations.
            n_generations: Number of utterances to generate per intent.
            n_hops: Number of iterations of generation-criticism mechanism.
            update_split: Whether to update the dataset split.

        Returns:
            List of generated samples.
        """
        original_split = dataset[split_name]
        new_samples = []
        for intent in dataset.intents:
            generated_utterances = self(
                intent_data=intent, n_generations=n_generations, n_hops=n_hops
            )
            new_samples.extend([
                {
                    Dataset.label_feature: intent.id,
                    Dataset.utterance_feature: utterance
                } for utterance in generated_utterances
            ])

        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([
                original_split, generated_split
            ])

        return [Sample(**sample) for sample in new_samples]


def _extract_utterances(response_text: str) -> list[str]:
    """Extract utterances from LLM output.

    Args:
        response_text: Response text from LLM.

    Returns:
        List of utterances.
    """
    raw_utterances = response_text.split("\n")
    res = [ut[ut.find(" ") + 1:] if " " in ut else ut for ut in raw_utterances]
    final = []
    for ut in res:
        words = ut.lower().split()
        if not ('feedback' in words or 'utterance' in words or
                'utterances' in words or 'AI' in words) and len(ut) > 0:
            final.append(ut)
    return final
