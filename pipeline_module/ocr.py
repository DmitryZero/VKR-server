import pytesseract
from PIL import Image


class RussianPDFOCR:
    def __init__(self, lang="rus", dpi=300):
        self.lang = lang
        self.dpi = dpi

    def recognize_pdf(self, input_pdf_doc):
        full_text = ""

        for idx, page in enumerate(input_pdf_doc):
            print(f"Распознаётся страница {idx + 1} из {len(input_pdf_doc)}...")

            # Создаём изображение с заданным DPI
            pix = page.get_pixmap(dpi=self.dpi)

            # Преобразуем Pixmap в изображение PIL
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Получаем текст с изображения с использованием pytesseract
            page_text = pytesseract.image_to_string(img, lang=self.lang)
            full_text += f"\n\n{page_text}"

        return full_text