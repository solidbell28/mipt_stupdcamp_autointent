# git clone https://github.com/voorhs/AutoIntent.git if an error occurs somewhere, it is better to remove '#', maybe it helps
# !cd AutoIntent
# !pip install autointent

import random
import re
from collections import defaultdict

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation import Generator
from autointent.generation.chat_templates import Message, Role
from autointent.generation.utterances import UtteranceGenerator
from autointent.schemas import Sample
from datasets import concatenate_datasets, Dataset as HFDataset

# new class

class AllIntentGenerator(UtteranceGenerator):
    def __init__(self, reasoning_generator: Generator, selection_generator: Generator):
        self.reasoning_generator = reasoning_generator  # To generate examples with explanations
        self.selection_generator = selection_generator  # To select the BEST answers

    def _build_class_level_prompt(
        self,
        examples: list[str],
        intent_name: str,
        other_intent: str,
        n_generate: int,
    ) -> Message:
        example_block = "\n".join(f"{i + 1}. {ex}" for i, ex in enumerate(examples))

        content = (
            f"Generate {n_generate} diverse, natural-sounding utterances for the intent "
            f"'{intent_name}'. Each example must clearly belong to '{intent_name}' and NOT to "
            f"'{other_intent}'. Follow these steps:\n\n"

            f"1. Examples of '{intent_name}':\n"
            f"{example_block}\n\n"

            f"2. Generate {n_generate} new examples numbered 1-{n_generate} using the EXACT format "
            f"below for every example:\n"
            f"   - Line 1: the utterance ONLY (no explanation or extra text).\n"
            f"   - Line 2: `Reasoning:` your explanation of why it fits '{intent_name}' and not "
            f"'{other_intent}'.\n\n"

            f"3. Analysis:\n"
            f"   - Identify the best example(s) (clearest fit for '{intent_name}' and farthest from "
            f"'{other_intent}').\n"
            f"   - Suggest which to choose for optimal distinctiveness.\n\n"

            f"Rules:\n"
            f"- Do NOT repeat provided examples.\n"
            f"- Utterance lines MUST contain ONLY the utterance text.\n"
            f"- Explanations MUST start on the next line with `Reasoning:`.\n"
            f"- Prioritize natural, human-like phrasing.\n"
            f"- Ensure clear contrast with '{other_intent}'."
        )
        return Message(role=Role.USER, content=content)
        return Message(role=Role.USER, content=content)

    def _build_critic_prompt(
        self,
        candidates: list[str],
        intent_name: str,
        other_intent_names: list[str], # doesn't use
        n_select: int,
    ) -> Message:
        options = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(candidates))
        plural = "utterances" if n_select > 1 else "utterance"
        content = (
            f"Select the {n_select} BEST {plural} for intent '{intent_name}'"
            f"Return ONLY the numbers of the selected {plural} separated by commas (e.g. '3,5,7').\n\n"
            f"Options:\n{options}"
        )
        return Message(role=Role.USER, content=content)

    def generate(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_final_per_class: int = 1,
        update_split: bool = True,
    ) -> list[Sample]:

        n_generate = n_final_per_class * 5 # 5 options for 1 necessary

        original_split = dataset[split_name]
        id_to_name = {intent.id: intent.name for intent in dataset.intents}
        utterances_by_intent = defaultdict(list)

        for sample in original_split:
            utterances_by_intent[sample["label"]].append(sample["utterance"])

        new_samples = []

        for intent_id, examples in utterances_by_intent.items():
            intent_name = id_to_name[intent_id]
            other_intent_names = [id_to_name[i] for i in id_to_name if i != intent_id]
            other_intent = random.choice(other_intent_names)
            few_shots = random.sample(examples, min(5, len(examples)))

            print(f"\n Generating for intent: {intent_name} (id={intent_id})")

            reasoning_prompt = self._build_class_level_prompt(
                few_shots, intent_name, other_intent, n_generate
            )
            raw_response = self.reasoning_generator.get_chat_completion([reasoning_prompt])
            candidates = self._extract_utterances(raw_response)

            # Сhoosing the best n_final_per_class
            if len(candidates) < n_final_per_class:
                selected_ids = list(range(1, len(candidates) + 1))
            else:
                critic_prompt = self._build_critic_prompt(
                    candidates,
                    intent_name,
                    other_intent_names,
                    n_select=n_final_per_class,
                )
                critic_response = self.selection_generator.get_chat_completion([critic_prompt])
                selected_ids = self._parse_indices_from_critic_response(
                    critic_response, max_id=len(candidates), n_select=n_final_per_class
                )

                if len(selected_ids) < n_final_per_class:
                    remaining = [
                        i for i in range(1, len(candidates) + 1) if i not in selected_ids
                    ][: n_final_per_class - len(selected_ids)]
                    selected_ids.extend(remaining)

            for idx in selected_ids:
                new_samples.append(
                    {
                        Dataset.label_feature: intent_id,
                        Dataset.utterance_feature: candidates[idx - 1],
                    }
                )

        if update_split and new_samples:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return [Sample(**sample) for sample in new_samples]

    def _extract_utterances(self, response_text: str) -> list[str]:
        utterances = []
        for line in response_text.strip().split("\n"):
            match = re.match(r"^\s*\d+\.\s+(.*)", line)
            if match:
                utterances.append(match.group(1).strip())
        return utterances

    def _parse_indices_from_critic_response(
        self, response: str, max_id: int, n_select: int
    ) -> list[int]:
        matches = re.findall(r"\d+", response)
        result: list[int] = []
        for m in matches:
            idx = int(m)
            if 1 <= idx <= max_id and idx not in result:
                result.append(idx)
            if len(result) == n_select:
                break
        return result
