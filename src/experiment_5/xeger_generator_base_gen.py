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


class RegexGeneratorBase:
    """
    A class-level utterance generator that uses regex-based synthesis and clustering
    to produce diverse and human-like paraphrases for user intents.

    The process involves:
    1. Generating regex patterns using a template-based prompt.
    2. Sampling utterances from regexes using Xeger.
    3. Filtering for uniqueness using Levenshtein similarity.
    4. Selecting representative examples via KMeans clustering.
    """

    def __init__(self, generator: Generator, model, template=ReSynthesizerTemplateEn, nums_for_reg=150, nums_reg=5):
        """
        Initialize the generator.

        Args:
            generator: A text generation model interface (e.g., OpenAI GPT wrapper).
            model: A SentenceTransformer model for embedding computation.
            template: template for generate regex
            nums_for_reg (int): Number of utterances to sample per regex.
            nums_reg (int): Number of regexes to generate per class.
        """
        self.generator = generator
        self.model = model
        self.template = template
        self.nums_for_reg = nums_for_reg
        self.nums_reg = nums_reg

    def _result_for_one_class(self, texts: list, num_clusters: int):
        """
        Cluster input utterances and return one representative per cluster.

        Args:
            texts (list): List of candidate utterances.
            num_clusters (int): Number of clusters to generate.

        Returns:
            list[str]: Cluster centroids represented by actual utterances.
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        kmeans = KMeans(n_clusters=num_clusters, random_state=1303)
        kmeans.fit_predict(embeddings.cpu().numpy())
        centroids = kmeans.cluster_centers_
        similarities = cosine_similarity(centroids, embeddings.cpu().numpy())
        closest_indices = similarities.argmax(axis=1)
        centroid_texts = [texts[i] for i in closest_indices]
        return centroid_texts

    def extract_xeger_regexes(self, response: str) -> list[str]:
        """
        Extract regexes from a numbered list in a text response.

        Args:
            response (str): Model response containing regex list.

        Returns:
            list[str]: Extracted regex strings.
        """
        lines = response.strip().split("\n")
        return [re.match(r"^\s*\d+\.\s*(.+)$", line).group(1).strip()
                for line in lines if re.match(r"^\s*\d+\.\s*(.+)$", line)]

    def _filter_similar_levenshtein(self, sentences: list[str], threshold: float = 0.9) -> list[str]:
        """
        Remove similar utterances using Levenshtein ratio.

        Args:
            sentences (list[str]): List of generated utterances.
            threshold (float): Similarity threshold above which to filter out duplicates.

        Returns:
            list[str]: Filtered list of diverse utterances.
        """
        unique = []
        for sentence in sentences:
            if all(Levenshtein.ratio(sentence, other) < threshold for other in unique):
                unique.append(sentence)
        return unique

    def generate_sentences_from_regex(self, regexes: list[str], n_per_regex: int = 30) -> list[str]:
        """
        Generate utterances by sampling from regexes using Xeger.

        Args:
            regexes (list[str]): List of regex strings.
            n_per_regex (int): Number of samples per regex.

        Returns:
            list[str]: Sampled and filtered utterances.
        """
        x = Xeger()
        sentences = []
        for regex in regexes:
            try:
                generated = [x.xeger(regex) for _ in range(n_per_regex)]
                sentences.extend(generated)
            except Exception as e:
                print(f"⚠️ Failed on regex: {regex}\n{e}")

        filtered = self._filter_similar_levenshtein(sentences)
        return filtered

    def extract_numbered_list(self, response: str) -> list[str]:
        """
        Parse a numbered list from a string into a list of text items.

        Args:
            response (str): Text containing numbered list (e.g., "1. Hello\n2. Goodbye").

        Returns:
            list[str]: Extracted text entries from the list.
        """
        lines = response.strip().split("\n")
        return [re.match(r"^\s*\d+\.\s*(.+)$", line).group(1).strip()
                for line in lines if re.match(r"^\s*\d+\.\s*(.+)$", line)]

    def expand_intent_data_for_class(self, intent: str, regexes: list[str], n_clusters: int) -> list[str]:
        """
        Generate and cluster utterances for a specific intent using regexes.

        Args:
            intent (str): The intent name (used only for display/logging).
            regexes (list[str]): Regex patterns to generate utterances from.
            n_clusters (int): Number of final utterances to return.

        Returns:
            list[str]: Final generated utterances for the intent.
        """
        generated = self.generate_sentences_from_regex(regexes, n_per_regex=self.nums_for_reg)
        selected = self._result_for_one_class(texts=generated, num_clusters=n_clusters)
        return selected

    def augment(self, dataset: Dataset, split_name: str = Split.TRAIN, n_generations: int = 3,
                update_split: bool = True) -> list[dict]:
        """
        Augment the dataset with synthetic utterances for each intent class.

        Args:
            dataset (Dataset): The dataset to augment.
            split_name (str): Name of the split to augment (e.g., 'train').
            n_generations (int): Number of final utterances per intent.
            update_split (bool): Whether to update the dataset object in-place.

        Returns:
            list[dict]: List of new samples with keys 'label' and 'utterance'.
        """
        original_split = dataset[split_name]
        id_to_name = {intent.id: intent.name for intent in dataset.intents}
        utterances_by_intent = defaultdict(list)

        for sample in original_split:
            utterances_by_intent[sample["label"]].append(sample["utterance"])

        new_samples = []
        augmenter = UtteranceGenerator(self.generator, prompt_maker=self.template(dataset))
        data_exsp = augmenter.augment(dataset, n_generations=self.nums_reg, update_split=False)
        exp_by_intent = defaultdict(list)
        for sample in data_exsp:
            exp_by_intent[sample.label].append(sample.utterance)

        for intent_id, examples in utterances_by_intent.items():
            intent_name = id_to_name[intent_id]
            generated_utterances = self.expand_intent_data_for_class(
                intent=intent_name,
                regexes=exp_by_intent[intent_id],
                n_clusters=n_generations
            )
            for utterance in generated_utterances:
                new_samples.append({
                    Dataset.label_feature: intent_id,
                    Dataset.utterance_feature: utterance
                })

        if update_split and new_samples:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return new_samples
