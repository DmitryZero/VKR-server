import asyncio
import os
import ssl

import json
import aiohttp
import fitz
from app.celery_folder.celery_worker import celery_app
from pipeline_module.interfaces import OutputWorkerData, InputApiData
from pipeline_module.pipeline import TextProcessingPipeline

@celery_app.task
def add(x, y):
    return x + y


@celery_app.task
def get_key_phrases(file_path: str, input_data_dict: dict):
    async def async_main():
        try:
            input_obj = InputApiData(**input_data_dict)
            pipeline = TextProcessingPipeline("ru_core_news_md", "paraphrase-multilingual-MiniLM-L12-v2")

            with fitz.open(file_path) as doc:
                result = pipeline.process_text(doc, input_obj.config)
                output_result = OutputWorkerData(input_data=input_obj, output_data=result)

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            payload = {
                "status": "done",
                "result": output_result.model_dump()
            }
            json_payload = json.dumps(payload, ensure_ascii=False)

            async with aiohttp.ClientSession() as session:
                await session.post(
                    input_obj.callback_url,
                    data=json_payload,
                    headers={"Content-Type": "application/json"},
                    ssl=ssl_context
                )

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    asyncio.run(async_main())
