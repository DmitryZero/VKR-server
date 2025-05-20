from typing import List

from rapidfuzz import fuzz


def filter_ner(
    input_ner_list,
    input_threshold: int = 90,
    filter_type: list[str] = ['PER'],
    max_ents: int = 10
) -> List[str]:
    # Фильтруем по типу и длине
    without_persons_ents = [
        ent for ent in input_ner_list
        if ent.label_ not in filter_type and len(ent.text) > 2
    ]

    # Удаляем дубликаты по смыслу
    unique_ents = []
    for ent in without_persons_ents:
        text = ent.text.strip().lower()
        is_duplicate = False
        for existing in unique_ents:
            if fuzz.partial_ratio(text, existing.text.strip().lower()) >= input_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_ents.append(ent)

    # Ограничиваем количество
    truncated_list = unique_ents[:max_ents]
    return [item.text for item in truncated_list]