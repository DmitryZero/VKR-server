from typing import List

from keyphrase_vectorizers import KeyphraseCountVectorizer
from spacy import Language

from pipeline_module.interfaces import FoundPhrases, PatternConfig


class PhraseCountVectorizerWrapper:
    @staticmethod
    def print_pretty_phrases(results: List[FoundPhrases]):
        for found_phrase in results:
            config = found_phrase.pattern_config
            current_phrases = found_phrase.found_words
            print(f"\n🔹{config.name} ({config.pattern})")
            for i, phrase in enumerate(current_phrases, 1):
                print(f"  {i:2d}. {phrase}")

    def __init__(self, nlp_model):
        self.nlp: Language = nlp_model

    def get_key_phrases(self, input_text: str, current_patterns: List[PatternConfig]) -> List[FoundPhrases]:
        results: List[FoundPhrases] = []
        for pattern_obj in current_patterns:
            current_vectorizer = KeyphraseCountVectorizer(
                spacy_pipeline=self.nlp,
                pos_pattern=pattern_obj.pattern,
                lowercase=True
            )
            try:
                current_vectorizer.fit_transform([input_text])
                current_phrases = current_vectorizer.get_feature_names_out()
                results.append(FoundPhrases(pattern_obj, current_phrases))
            except ValueError as e:
                print(f"Паттерн '{pattern_obj.name}' не нашёл фраз: {e}")
                results.append(FoundPhrases(pattern_obj, []))
        return results
