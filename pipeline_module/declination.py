from typing import List, Optional

from pymorphy3 import MorphAnalyzer
from spacy.tokens import Doc

from pipeline_module.interfaces import TokenInfo, POS_MAPPING


class TextDeclinationObj:
    @staticmethod
    def print_pretty_phrases(
            original_phrases: List[str],
            declined_phrases: List[str],
            description: str = "Сравнение до и после"
    ) -> None:
        if len(original_phrases) != len(declined_phrases):
            print("Ошибка: списки фраз разной длины")
            return

        print(f"\n🔹 {description}")

        # Выводим фразы парами
        for idx, orig in enumerate(original_phrases):
            print(f"{idx + 1:2d}. {orig} -> {declined_phrases[idx]}")

        print('-' * 60)

    def __init__(self, inner_processed_nlp_text: Doc):
        self.morph = MorphAnalyzer()
        self.tokens_info: List[TokenInfo] = []

        for token in inner_processed_nlp_text:
            gender_list = token.morph.get("Gender")
            number_list = token.morph.get("Number")

            current_gender = gender_list[0].lower() if gender_list else None
            current_number = number_list[0].lower() if number_list else None

            self.tokens_info.append(TokenInfo(
                word=token.text,
                pos=token.pos_,
                start_char=token.idx,
                end_char=token.idx + len(token),
                number=current_number,
                gender=current_gender
            ))

    def __inflect_word_to_nominative(self, input_word: str, input_pos: str, input_gender: Optional[str] = None,
                                     input_number: Optional[str] = None) -> str:
        parsed_info = self.morph.parse(input_word)
        if not parsed_info:
            return input_word

        def is_suitable(variant):
            mapped_pos = POS_MAPPING.get(input_pos)
            if mapped_pos is None:
                return False
            if variant.tag.POS != mapped_pos:
                return False
            if input_gender and 'GNdr' not in variant.tag:
                if input_gender == "masc" and 'masc' not in variant.tag:
                    return False
                if input_gender == "fem" and 'femn' not in variant.tag:
                    return False
                if input_gender == "neut" and 'neut' not in variant.tag:
                    return False
            if input_number:
                if input_number == "sing" and 'sing' not in variant.tag:
                    return False
                if input_number == "plur" and 'plur' not in variant.tag:
                    return False
            return True

        best_match = next((p for p in parsed_info if is_suitable(p)), None)
        if best_match is None and len(parsed_info) == 1:
            best_match = parsed_info[0]

        if best_match:
            inflected_word = best_match.inflect({'nomn'})
            if inflected_word:
                return inflected_word.word

        return input_word

    def __decline_phrase_to_nominative(self, current_phrase: str, preserve_case: bool) -> str:
        """Склоняет слова в фразе в именительный падеж, учитывая предыдущее слово."""
        words_in_phrase = current_phrase.split()
        declined_words = []
        previous_pos = None
        phrase_has_adp = False

        for word_token in words_in_phrase:
            matched_token = next((t for t in self.tokens_info if t.word == word_token), None)

            # Если прошлое слово было предлогом или существительным — текущее не склоняем
            if previous_pos in ("PREP", "NOUN", "PROPN", "ADP") or phrase_has_adp:
                declined_words.append(word_token)
            else:
                if matched_token:
                    phrase_was_declined = True
                    declined_word = self.__inflect_word_to_nominative(
                        matched_token.word,
                        matched_token.pos,
                        matched_token.gender,
                        matched_token.number
                    )

                    if declined_word and preserve_case:
                        restored_word = []
                        for orig_char, new_char in zip(word_token, declined_word):
                            if orig_char.isupper():
                                restored_word.append(new_char.upper())
                            else:
                                restored_word.append(new_char.lower())
                        # Добавляем оставшиеся символы (если новая форма длиннее)
                        if len(declined_word) > len(word_token):
                            restored_word.extend(declined_word[len(word_token):].lower())
                        declined_word = ''.join(restored_word)

                    declined_words.append(declined_word)
                else:
                    declined_words.append(word_token)

            # Запоминаем POS текущего слова для следующего шага
            if matched_token:
                previous_pos = matched_token.pos
                if previous_pos == 'ADP':
                    phrase_has_adp = True
            else:
                previous_pos = None

        return " ".join(declined_words)

    def decline_phrase_list(self, input_data: list[str], preserve_case: bool = True) -> List[str]:
        final_result: List[str] = []
        for phrase_item in input_data:
            current_declined_phrase = self.__decline_phrase_to_nominative(phrase_item, preserve_case)
            final_result.append(current_declined_phrase)

        return final_result
