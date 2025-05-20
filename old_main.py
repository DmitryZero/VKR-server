import tempfile
from contextlib import asynccontextmanager
import fitz

import aiofiles
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uuid, os, aiohttp

from pipeline_module.interfaces import InputPipelineData
from pipeline_module.pipeline import TextProcessingPipeline

from pydantic_settings import BaseSettings

class InputData(BaseModel):
    file_link: str
    callback_url: str
    config: InputPipelineData


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при запуске
    app.state.my_instance = TextProcessingPipeline("ru_core_news_md", "paraphrase-multilingual-MiniLM-L12-v2")
    yield

app = FastAPI(lifespan=lifespan)

# === Модели ===
class FileLink(BaseModel):
    url: str
    callback_url: str

class UploadWithCallback(BaseModel):
    callback_url: str

@app.get("/")
def test(request: Request):
    return {"Сервер запущен, pipeline объект": str(request.app.state.my_instance)}

# === Загрузка по ссылке ===
@app.post("/get_keyphrases_from_url/")
async def upload_from_url(input_item: InputData, request: Request):
    task_id = str(uuid.uuid4())

    # Создаём временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_path = tmp_file.name

    # Скачиваем и сохраняем файл
    async with aiohttp.ClientSession() as session:
        async with session.get(input_item.file_link) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=400, detail="Failed to download file")
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(await resp.read())

    try:
        content_type = resp.headers.get("Content-Type")
        if not content_type or "pdf" not in content_type.lower():
            raise HTTPException(status_code=400, detail="URL does not point to a PDF file")

        # Чтение и обработка PDF
        with fitz.open(tmp_path) as document:
            print(str(document))

            pipeline: TextProcessingPipeline = request.app.state.my_instance
            result = pipeline.process_text(document)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Отправка результатов
    async with aiohttp.ClientSession() as session:
        print({"status": "done", "keywords_obj": result})
        await session.post(input_item.callback_url, json={"status": "done", "keywords_obj": result})

    return {"task_id": task_id}
