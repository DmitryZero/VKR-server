from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel


@dataclass
class PatternConfig:
    name: str
    code: str
    pattern: str
    diversity: float = 0.5
    top_n: int = 5
    threshold_filter: float = 0.2

@dataclass
class FoundPhrases:
    pattern_config: PatternConfig
    found_words: list[str]

@dataclass
class KeyPhraseData:
    key_phrase: str
    value: float

@dataclass
class BertKeyPhrases:
    pattern_config: PatternConfig
    found_key_phrases: list[KeyPhraseData]

@dataclass
class TokenInfo:
    word: str
    pos: str
    start_char: int
    end_char: int
    number: Optional[str] = None
    gender: Optional[str] = None

@dataclass
class NerConfig:
    phrase_amount: int
    exclude_types: Optional[list[str]]
    input_threshold: int

@dataclass
class InputPipelineData:
    phrases_config: list[PatternConfig]
    ner_config: NerConfig

@dataclass
class PatternKeyPhrases:
    pattern_config: PatternConfig
    key_phrases: list[str]

@dataclass
class OutputPipelineData:
    key_phrases_obj: list[PatternKeyPhrases]
    ner_phrases: list[str]

class InputApiData(BaseModel):
    file_link: str
    callback_url: str
    config: InputPipelineData

class OutputWorkerData(BaseModel):
    input_data: InputApiData
    output_data: OutputPipelineData

POS_MAPPING = {
    # Существительные
    "NOUN": "NOUN",  # существительное (дом)
    "PROPN": "NOUN",  # имя собственное (Москва)

    # Прилагательные
    "ADJ": "ADJF",  # прилагательное (красный)
    "DET": "ADJF",  # определитель (этот, тот)
    "NUM": "NUMR",  # числительное (пять, пятый)

    # Местоимения
    "PRON": "NPRO",  # местоимение (я, он, свой)

    # Глаголы и причастия
    "VERB": "VERB",  # глагол (писать)
    "AUX": "VERB",  # вспомогательный глагол (быть)
    "PART": "PRCL",  # частица (бы, же, ли)

    # Наречия
    "ADV": "ADVB",  # наречие (быстро)
    "ADP": "PREP",  # предлог (в, на)
    "CCONJ": "CONJ",  # союз (и, а)
    "SCONJ": "CONJ",  # подчинительный союз (что, чтобы)

    # Другие
    "INTJ": "INTJ",  # междометие (ой)
    "SYM": None,  # символ ($, %)
    "PUNCT": None,  # пунктуация
    "X": None,  # другое
    "SPACE": None,  # пробел
}