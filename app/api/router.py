import os
import uuid

import aiohttp
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException

from app.celery_folder.celery_worker import celery_app
from app.config import settings, redis_client
from app.celery_folder.tasks import add, get_key_phrases
from pipeline_module.interfaces import InputPipelineData, InputApiData

router = APIRouter(tags=['API'])

@router.get("/api/test/")
async def test():
    task = add.apply_async(args=(1, 5), countdown=5)  # Запуск через 5 сек
    return {"task_id": task.id}


@router.get("/api/result/{task_id}")
async def get_result(task_id: str):
    result = AsyncResult(task_id, app=celery_app)

    return {
        "task_id": task_id,
        "status": result.status,  # 'PENDING', 'STARTED', 'SUCCESS', etc.
        "result": result.result  # None, если ещё не готов
    }

@router.post("/api/send_request_to_get_key_phrases/")
async def send_request_to_get_key_phrases(input_data: InputApiData):
    print("input_data")
    print(input_data)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(input_data.file_link) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail="Не удалось скачать файл")

                file_content = await resp.read()

        if len(file_content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Превышен максимальный размер файла (300 МБ).")

        upload_dir = settings.UPLOAD_DIR
        total_size = sum(os.path.getsize(os.path.join(upload_dir, f)) for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f)))
        if total_size + len(file_content) > settings.MAX_DIR_SIZE_GB * 1024 * 1024 * 1024:
            raise HTTPException(status_code=507, detail="Превышен общий лимит размера файлов (20 ГБ). Освободите место и повторите попытку.")

        filename = f"{uuid.uuid4()}.pdf"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file_content)

        get_key_phrases.delay(file_path, input_data.model_dump())
        return {"status": "processing started"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки файла: {str(e)}")