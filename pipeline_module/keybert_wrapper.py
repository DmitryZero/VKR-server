from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline_module.interfaces import BertKeyPhrases, FoundPhrases, KeyPhraseData


class CustomKeyBertForArchive:
    @staticmethod
    def pretty_print_bert_output(output_data: list[BertKeyPhrases]):
        for output in output_data:
            print(
                f"Паттерн: {output.pattern_config.name} {output.pattern_config.code} ({output.pattern_config.pattern})")
            print(
                f"  Топ {output.pattern_config.top_n} фраз (Diversity: {output.pattern_config.diversity}, Threshold: {output.pattern_config.threshold_filter}):")

            # Исправлено: обращаемся к атрибутам KeyPhraseData
            for phrase_data in output.found_key_phrases:
                print(f"    {phrase_data.key_phrase}: {phrase_data.value:.4f}")

            print('-' * 40)

    def __init__(
            self,
            bert_model_name: str
    ) -> None:
        self.model: SentenceTransformer = SentenceTransformer(bert_model_name)

    def extract_keywords(
            self,
            doc_text: str,
            phrases_list: list[FoundPhrases]
    ) -> List[BertKeyPhrases]:
        doc_embedding: np.ndarray = self.model.encode([doc_text])[0]

        print("phrases_list", phrases_list, sep=" ")
        output_data = []
        for idx, phrase_obj in enumerate(phrases_list):
            pattern_config = phrase_obj.pattern_config
            phrases = phrase_obj.found_words

            diversity = pattern_config.diversity
            top_number = pattern_config.top_n

            # Получаем эмбеддинги для всех кандидатов
            candidate_embeddings: np.ndarray = self.model.encode(phrases)

            top_n = min(top_number, len(phrases))

            # Выбираем ключевые слова с помощью алгоритма MMR
            selected_keywords = CustomKeyBertForArchive.__mmr(
                document_embedding=doc_embedding,
                candidate_embeddings=candidate_embeddings,
                candidates=phrases,
                top_n=top_n,
                diversity=diversity,
                threshold_filter=pattern_config.threshold_filter,
            )

            # Создаём объект BertOutputData для текущего паттерна
            output_data.append(BertKeyPhrases(pattern_config, selected_keywords))

        return output_data

    @staticmethod
    def __mmr(
            document_embedding: np.ndarray,
            candidate_embeddings: np.ndarray,
            candidates: list[str],
            top_n: int,
            diversity: float,
            threshold_filter: float
    ) -> List[KeyPhraseData]:
        candidate_to_doc_similarity = cosine_similarity(candidate_embeddings, [document_embedding]).flatten()
        above_threshold = candidate_to_doc_similarity >= threshold_filter
        candidate_embeddings = candidate_embeddings[above_threshold]
        candidates = candidates[above_threshold]
        candidate_to_doc_similarity = candidate_to_doc_similarity[above_threshold]

        if len(candidates) == 0:
            return []

        # breakpoint()

        candidate_similarity_matrix = cosine_similarity(candidate_embeddings)

        selected_keywords: List[KeyPhraseData] = []
        selected_indices = []

        best_initial_idx = np.argmax(candidate_to_doc_similarity)

        selected_keywords.append(
            KeyPhraseData(candidates[best_initial_idx], round(candidate_to_doc_similarity[best_initial_idx], 4)))
        selected_indices.append(best_initial_idx)

        while len(selected_keywords) < top_n:
            candidates_mmr_scores = []

            for candidate_idx in range(len(candidates)):
                if candidate_idx in selected_indices:
                    continue

                relevance = candidate_to_doc_similarity[candidate_idx]

                similarity_to_selected = candidate_similarity_matrix[candidate_idx][selected_indices]
                max_similarity = max(similarity_to_selected) if selected_indices else 0

                mmr_score = (1 - diversity) * relevance - diversity * max_similarity
                print(mmr_score)

                candidates_mmr_scores.append((candidate_idx, mmr_score))

            if not candidates_mmr_scores:
                break

            print("candidates", candidates, sep=" ")
            print("candidates_mmr_scores", candidates_mmr_scores, sep=" ")
            best_candidate_idx, _ = max(candidates_mmr_scores, key=lambda x: x[1])

            selected_keywords.append(KeyPhraseData(candidates[best_candidate_idx],
                                                   round(candidate_to_doc_similarity[best_candidate_idx], 4)))
            selected_indices.append(best_candidate_idx)

        selected_keywords.sort(key=lambda x: x.value, reverse=True)

        return selected_keywords