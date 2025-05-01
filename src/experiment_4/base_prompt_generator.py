from typing import ClassVar, Optional, List
import random

from autointent.generation.chat_templates._evolution_templates_schemas import Message, Role
from autointent import Dataset
from autointent.custom_types import Split
from autointent.schemas import Intent


class BasePromptGeneratorTemplate:
    """Base chat template for using LLM-based prompt engineer."""

    _GENERATE_INITIAL_INSTRUCTION: ClassVar[str]
    _GENERATE_INSTRUCTION: ClassVar[str]

    def __init__(
        self, dataset: Dataset, split: str = Split.TRAIN,
        max_sample_utterances: Optional[int] = None
    ) -> None:
        """
        Initialize the BasePromptGeneratorTemplate.

        Args:
            dataset: Dataset to use for generating utterances.
            split: Dataset split to use for generating examples.
            max_sample_utterances: Maximum number of sample utterances to include.
        """
        self.dataset = dataset
        self.split = split
        self.max_sample_utterances = max_sample_utterances
        self.initial_examples = ""

    def __call__(
        self, intent_data: Intent, critic_text: Optional[str], n_examples: int
    ) -> List[Message]:
        """
        Generate a message to request a prompt.

        Args:
            intent_data: Intent data for which to generate examples.
            critic_text: Feedback from a critic (may not be provided).
            n_examples: Number of examples to generate.

        Returns:
            List of messages (consisting only of 1 message) for the chat template.
        """
        if critic_text is not None:
            new_message = self._generate_message(critic_text)
            return [new_message]

        in_domain_samples = self.dataset[self.split].filter(
            lambda sample: sample[Dataset.label_feature] is not None
        )
        if self.dataset.multilabel:
            def filter_fn(sample):
                return sample[Dataset.label_feature][intent_data.id] == 1
        else:
            def filter_fn(sample):
                return sample[Dataset.label_feature] == intent_data.id

        filtered_split = in_domain_samples.filter(filter_fn)
        sample_utterances = filtered_split[Dataset.utterance_feature]

        if self.max_sample_utterances is not None and len(sample_utterances) > self.max_sample_utterances:
            sample_utterances = random.sample(
                sample_utterances, k=self.max_sample_utterances
            )

        message = self._generate_initial_message(
            intent_data, sample_utterances, n_examples
        )
        return [message]

    def _generate_message(self, critic_text: str) -> Message:
        """
        Generate message for the chat template.

        Args:
            critic_text: Feedback from a critic.

        Returns:
            A message for the chat template.
        """
        content = self._GENERATE_INSTRUCTION.format(
            critic_text=critic_text, initial_examples=self.initial_examples
        )
        return Message(role=Role.USER, content=content)

    def _generate_initial_message(
        self, intent_data: Intent, samples: List[str], n_examples: int
    ) -> Message:
        """
        Generate initial message for the chat template.
        On the start of the prompt generation process no feedback from critic is provided.

        Args:
            intent_data: Intent data for which to generate examples.
            samples: Sample utterances to use in prompt.
            n_examples: Number of examples to generate.

        Returns:
            Initial message for the chat template.
        """
        initial_examples = ""
        for idx, sample in enumerate(samples):
            initial_examples += f"{idx + 1}. {sample} \n "
        self.initial_examples = initial_examples

        content = self._GENERATE_INITIAL_INSTRUCTION.format(
            intent_name=intent_data.name,
            initial_examples=initial_examples,
            n_examples=n_examples
        )
        return Message(role=Role.USER, content=content)
