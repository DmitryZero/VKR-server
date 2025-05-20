import spacy

from pipeline_module.declination import TextDeclinationObj
from pipeline_module.interfaces import InputPipelineData, OutputPipelineData, PatternKeyPhrases, PatternConfig, NerConfig, \
    FoundPhrases
from pipeline_module.keybert_wrapper import CustomKeyBertForArchive
from pipeline_module.ner import filter_ner
from pipeline_module.ocr import RussianPDFOCR
from pipeline_module.phrase_extractor import PhraseCountVectorizerWrapper


class TextProcessingPipeline:
    def __init__(self, spacy_model_name: str, bert_model_name: str):
        # Инициализация моделей
        self.spacy_nlp_model = spacy.load(spacy_model_name)
        self.spacy_nlp_model.max_length = 400000  # Увеличиваем максимальную длину текста
        self.bert_extractor = CustomKeyBertForArchive(bert_model_name)
        self.phrase_extractor = PhraseCountVectorizerWrapper(self.spacy_nlp_model)
        self.ocr = RussianPDFOCR()

    @staticmethod
    def get_default_config() -> InputPipelineData:
        patterns = [
            PatternConfig("Одиночные существительные", "one_noun", "<N.*>", 0.3, 10, 0.1),
            PatternConfig("Биграммы существительных", "bigramm_noun", "<N.*><ADP>?<N.*>", 0.4, 10, 0.3),
            PatternConfig("Биграммы прилагательных и существительных", "bigramm_adj_noun", "<ADJ><ADP>?<N.*>", 0.6, 10,
                          0.3),
            PatternConfig("Большие фразы", "long_phrases", "(<A.*>?<ADP>?<A.*>?<N.*>){3,}", 0.7, 15, 0.4),
        ]
        return InputPipelineData(
            phrases_config=patterns,
            ner_config=NerConfig(
                input_threshold=15,
                exclude_types=["PER"],
                phrase_amount=80
            )
        )

    def process_text(self, document_to_process, config: InputPipelineData = None) -> OutputPipelineData:
        if config is None:
            config = self.get_default_config()

        print(f'Распознавание документа')
        text_from_document = self.ocr.recognize_pdf(document_to_process)
        processed_through_nlp_text = self.spacy_nlp_model(text_from_document)

        # Извлечение фраз по паттернам
        print(f'Извлечение фраз по паттернам')
        found_phrases = self.phrase_extractor.get_key_phrases(text_from_document, config.phrases_config)
        print("found_phrases", found_phrases, sep=" ")

        # Извлечение ключевых фраз с помощью BERT
        print(f'Извлечение ключевых фраз с помощью BERT')
        key_phrases = self.bert_extractor.extract_keywords(text_from_document, found_phrases)

        # Извлечение NER-сущностей
        print(f'Извлечение NER-сущностей')
        total_ner_list = filter_ner(processed_through_nlp_text.ents, config.ner_config.input_threshold, config.ner_config.exclude_types, config.ner_config.phrase_amount)

        # Склонение фраз
        print(f'Склонение NER фраз')
        declination_obj = TextDeclinationObj(processed_through_nlp_text)
        declination_ner = declination_obj.decline_phrase_list(total_ner_list, preserve_case=True)

        # Склонение ключевых фраз
        print(f'Склонение ключевых фраз')
        declined_key_phrases: list[PatternKeyPhrases] = []
        for item in key_phrases:
            original_phrases = [phrase.key_phrase for phrase in item.found_key_phrases]
            declination_phrases = declination_obj.decline_phrase_list(original_phrases, preserve_case=False)
            declined_key_phrases.append(PatternKeyPhrases(
                item.pattern_config,
                declination_phrases
            ))

        print(f'Вывод результата')
        return OutputPipelineData(
            declined_key_phrases,
            declination_ner
        )

    @staticmethod
    def print_results(final_results: OutputPipelineData):
        """Красивый вывод результатов"""
        # Вывод ключевых фраз
        print("\n=== КЛЮЧЕВЫЕ ФРАЗЫ ===")
        for idx, item in enumerate(final_results.key_phrases_obj):
            config = item.pattern_config
            current_phrases = item.key_phrases
            print(f"\n🔹{config}")
            for j, phrase in enumerate(current_phrases, 1):
                print(f"  {j:2d}. {phrase}")

        # Вывод NER-сущностей
        print("\n=== NER-СУЩНОСТИ ===")
        for j, phrase in enumerate(final_results.ner_phrases, 1):
            print(f"  {j:2d}. {phrase}")

