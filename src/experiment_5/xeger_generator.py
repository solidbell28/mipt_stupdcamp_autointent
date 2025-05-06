import re
from collections import defaultdict
from xeger import Xeger
import Levenshtein
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset, concatenate_datasets
from regex_template import ReSynthesizerTemplateEn
from autointent.generation.utterances import UtteranceGenerator
from autointent.generation import Generator
from datasets import Dataset as HFDataset
from autointent.schemas import Sample


class XegerGenerator:
    """
    A generator that creates synthetic utterances for each intent class using xeger-based and
    semantic-based augmentation. It combines prompt engineering, regex pattern generation,
    Xeger-based sampling, filtering, and clustering to produce diverse training data.

    Attributes:
        generator (Generator): A text generation interface (e.g., OpenAI or similar).
        model (SentenceTransformer): A sentence embedding model.
        nums_reg (int): Number of regexes to generate per class.
        nums_for_reg (int): Number of utterances to sample per regex.
    """

    def __init__(self, generator: Generator, model, nums_reg=5, nums_for_reg=50):
        """
        Initialize the generator with a text generation model and an embedding model.
        Args:
            generator (Generator): A language model for regex and semantic generation.
            model (SentenceTransformer): A model to encode sentences into embeddings.
            nums_reg (int): Number of regex patterns to generate per intent.
            nums_for_reg (int): Number of utterances to generate from each regex.
        """
        self.generator = generator
        self.model = model
        self.nums_reg = nums_reg
        self.nums_for_reg = nums_for_reg
        self.xeger = Xeger()

    def _build_semantic_prompt(self, examples: List[str], intent_name: str) -> Message:
        """
        Build a prompt asking the model to generate diverse paraphrases for a given intent.

        Args:
            examples (List[str]): Example utterances.
            intent_name (str): Name of the intent.

        Returns:
            Message: A prompt formatted for the LLM.
        """
        example_block = "\n".join(f"- {ex}" for ex in examples)
        return Message(
            role=Role.USER,
            content=(
                f"Generate exactly 15 diverse user utterances for intent: '{intent_name}'\n"
                f"Examples:\n{example_block}\n\n"
                f"Rules:\n"
                f"- Create completely new variations\n"
                f"- Include different phrasing styles (questions, commands, etc.)\n"
                f"- Add greetings, polite forms, and natural variations\n"
                f"- Output exactly 15 numbered items\n"
                f"- No explanations"
            )
        )

    def _build_regex_prompt(self, examples: List[str], intent_name: str) -> Message:
        """
        Build a prompt asking the model to generate regex patterns for an intent.

        Args:
            examples (List[str]): Example utterances for regex derivation.
            intent_name (str): Name of the intent.

        Returns:
            Message: Prompt to generate regex patterns.
        """
        example_block = "\n".join(f"- {ex}" for ex in examples)
        return Message(
            role=Role.USER,
            content=(
                f"Create 10 diverse regex patterns for intent: '{intent_name}'\n"
                f"Based on:\n{example_block}\n\n"
                f"Constraints:\n"
                f"- Use semantic placeholders like (GENRE), (ARTIST)\n"
                f"- Include optional greetings/polite forms\n"
                f"- Cover different phrasing styles\n"
                f"- Each should generate 50+ variants\n"
                f"- Output numbered regexes only"
            )
        )

    def _extract_numbered_list(self, response: str) -> List[str]:
        """
        Parse a numbered list from the LLM response.

        Args:
            response (str): Text output from the LLM.

        Returns:
            List[str]: Parsed list items.
        """
        return [
            re.split(r"^\d+\.\s*", line, maxsplit=1)[1].strip()
            for line in response.strip().split("\n")
            if re.match(r"^\d+\.", line)
        ]

    def _generate_from_regex(self, regexes: List[str], n_per_regex: int = 50) -> List[str]:
        """
        Generate utterances by sampling from regex patterns using Xeger.

        Args:
            regexes (List[str]): List of regex strings.
            n_per_regex (int): Number of samples per regex.

        Returns:
            List[str]: Generated utterances.
        """
        utterances = []
        for regex in regexes:
            try:
                utterances.extend([self.xeger.xeger(regex) for _ in range(n_per_regex)])
            except Exception as e:
                print(f"Regex error: {regex} - {str(e)}")
        return list(set(utterances))

    def _cluster_utterances(self, texts: List[str], n_clusters: int) -> List[str]:
        """
        Cluster utterances and return one representative per cluster (closest to centroid).

        Args:
            texts (List[str]): List of utterances.
            n_clusters (int): Number of clusters.

        Returns:
            List[str]: Cluster centroids as utterances.
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit_predict(embeddings.cpu().numpy())
        centroids = kmeans.cluster_centers_
        sim_matrix = cosine_similarity(centroids, embeddings.cpu().numpy())
        closest_indices = sim_matrix.argmax(axis=1)

        return [texts[i] for i in closest_indices]

    def _filter_similar(self, utterances: List[str], threshold: float = 0.9) -> List[str]:
        """
        Filter out near-duplicate utterances using Levenshtein similarity.

        Args:
            utterances (List[str]): List of candidate utterances.
            threshold (float): Maximum allowed similarity.

        Returns:
            List[str]: Filtered list of diverse utterances.
        """
        filtered = []
        for utt in utterances:
            if not any(Levenshtein.ratio(utt.lower(), f.lower()) >= threshold for f in filtered):
                filtered.append(utt)
        return filtered

    def generate(self, dataset: Dataset, split_name: str = Split.TRAIN,
                 n_generations: int = 3, update_split: bool = True) -> List[Sample]:
        """
        Generate synthetic data for each intent in the dataset.

        Args:
            dataset (Dataset): The HuggingFace-style dataset to augment.
            split_name (str): Name of the split to augment (e.g., 'train').
            n_generations (int): Number of final synthetic samples per intent.
            update_split (bool): Whether to modify the dataset in-place.

        Returns:
            List[Sample]: List of generated samples.
        """
        original_split = dataset[split_name]
        id_to_name = {intent.id: intent.name for intent in dataset.intents}
        utterances_by_intent = defaultdict(list)

        for sample in original_split:
            utterances_by_intent[sample["label"]].append(sample["utterance"])

        new_samples = []

        for intent_id, examples in utterances_by_intent.items():
            intent_name = id_to_name[intent_id]
            few_shots = random.sample(examples, min(5, len(examples)))

            semantic_prompt = self._build_semantic_prompt(few_shots, intent_name)
            semantic_response = self.generator.get_chat_completion([semantic_prompt])
            semantic_examples = self._extract_numbered_list(semantic_response)

            regex_prompt = self._build_regex_prompt(semantic_examples, intent_name)
            regex_response = self.generator.get_chat_completion([regex_prompt])
            regexes = self._extract_numbered_list(regex_response)

            generated = self._generate_from_regex(regexes, n_per_regex=50)
            generated = self._filter_similar(generated)
            selected = self._cluster_utterances(generated, n_clusters=n_generations)

            for utterance in selected:
                new_samples.append({
                    Dataset.label_feature: intent_id,
                    Dataset.utterance_feature: utterance
                })

        if update_split and new_samples:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return [Sample(**sample) for sample in new_samples]
