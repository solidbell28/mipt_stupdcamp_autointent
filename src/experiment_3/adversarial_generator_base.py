from autointent.generation import Generator
from autointent import Dataset
from autointent.generation.chat_templates._evolution_templates_schemas import Message, Role
from autointent.schemas import Intent, Sample
from autointent.custom_types import Split

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from collections import defaultdict
import re


class AdversarialClassLevelGeneratorBase:
    """
    A base class for class-level utterance generation using prompting and internal critique.
    It generates multiple candidate utterances per intent and selects the most human-like ones.
    """

    def __init__(self, generator: Generator):
        """
        Initialize the generator.

        Args:
            generator: An LLM-based text generator used for both generation and internal critique.
        """
        self.generator = generator

    def _build_class_level_prompt(self, examples: list[str], intent_name: str, n_generate: int) -> Message:
        """
        Build a prompt asking the LLM to generate diverse utterances for a given intent.

        Args:
            examples: Few-shot example utterances for the intent.
            intent_name: The name of the intent.
            n_generate: Number of new utterances to request.

        Returns:
            Message: A prompt that instructs the model to generate a list of new utterances.
        """
        example_block = "\n".join(f"{i + 1}. {ex}" for i, ex in enumerate(examples))
        content = (
            f"You are generating paraphrased user utterances for the intent: '{intent_name}'.\n\n"
            f"they should cover the maximum number of situations that fit this intent."
            f"Here are some examples of how users typically phrase this intent:\n"
            f"{example_block}\n\n"
            f"Please generate {n_generate} new, diverse, natural-sounding utterances with the same intent.\n"
            f"Make sure they do not repeat the above and sound as if written by different humans.\n"
            f"Output as a numbered list."
        )
        return Message(role=Role.USER, content=content)

    def _build_critic_prompt(self, candidates: list[str], intent_name: str, n_best: int = 3) -> Message:
        """
        Build a prompt instructing the model to select the most natural utterances from a list.

        Args:
            candidates: List of generated utterances.
            intent_name: The intent these utterances correspond to.
            n_best: Number of best utterances to select.

        Returns:
            Message: A critic prompt that asks for the top-N most human-like utterances.
        """
        options = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(candidates))
        content = (
            f"You are a critic. You are given multiple paraphrased user utterances for intent: '{intent_name}'.\n\n"
            f"Select the {n_best} most natural and human-like ones.\n"
            f"List only their numbers separated by commas (e.g., 1, 3, 4). DO NOT explain.\n\n"
            f"Candidates:\n{options}"
        )
        return Message(role=Role.USER, content=content)

    def augment(
            self,
            dataset: Dataset,
            split_name: str = Split.TRAIN,
            n_generations: int = 3,
            update_split: bool = True
    ) -> list[Sample]:
        """
        Augment a dataset by generating and filtering new utterances for each intent.

        Args:
            dataset: Dataset to augment.
            split_name: Dataset split for which to apply augmentations.
            n_generations: Number of utterances to generate per intent.
            update_split: Whether to modify the dataset in-place by adding the new utterances.

        Returns:
            list[Sample]: List of newly generated and selected samples.
        """
        n_generate = n_generations * 2
        original_split = dataset[split_name]
        id_to_name = {intent.id: intent.name for intent in dataset.intents}
        utterances_by_intent = defaultdict(list)

        for sample in original_split:
            utterances_by_intent[sample["label"]].append(sample["utterance"])

        new_samples = []

        for intent_id, examples in utterances_by_intent.items():
            intent_name = id_to_name[intent_id]
            few_shots = random.sample(examples, min(5, len(examples)))

            prompt = self._build_class_level_prompt(few_shots, intent_name, n_generate)
            raw_response = self.generator.get_chat_completion([prompt])
            candidates = self._extract_utterances(raw_response)

            critic_prompt = self._build_critic_prompt(candidates, intent_name, n_best=n_generations)
            critic_response = self.generator.get_chat_completion([critic_prompt])
            selected_ids = self._parse_indices_from_critic_response(critic_response, max_id=len(candidates))
            selected = [candidates[i - 1] for i in selected_ids]

            for ut in selected:
                new_samples.append({
                    Dataset.label_feature: intent_id,
                    Dataset.utterance_feature: ut
                })

        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return [Sample(**sample) for sample in new_samples]

    def _extract_utterances(self, response_text: str) -> list[str]:
        """
        Extract utterances from a numbered list in the model's raw response.

        Args:
            response_text: Text returned by the generation LLM.

        Returns:
            list[str]: Cleaned list of extracted utterances.
        """
        lines = response_text.strip().split("\n")
        utterances = []
        for line in lines:
            if line and line[0].isdigit() and '.' in line:
                try:
                    _, text = line.split('.', 1)
                    utterances.append(text.strip())
                except ValueError:
                    continue
        return utterances

    def _parse_indices_from_critic_response(self, response: str, max_id: int) -> list[int]:
        """
        Parse selected utterance indices from critic model response.

        Args:
            response: Text response from the critic.
            max_id: Maximum valid index (based on number of candidates).

        Returns:
            list[int]: List of selected 1-based indices.
        """
        matches = re.findall(r"\d+", response)
        return [int(m) for m in matches if 1 <= int(m) <= max_id]
