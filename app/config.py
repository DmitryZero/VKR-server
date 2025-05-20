import os
import ssl
import redis
from celery import Celery
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_HOST: str
    BASE_URL: str
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    REDIS_URL: str
    UPLOAD_DIR: str = os.path.join(BASE_DIR, 'app/uploads')
    MAX_FILE_SIZE_MB: int
    MAX_DIR_SIZE_GB: int
    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/.env")


# Получаем параметры для загрузки переменных среды
settings = Settings()

ssl_options = {
    "ssl_cert_reqs": ssl.CERT_NONE,
    "ssl_check_hostname": False  # это важно!
}

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    password=settings.REDIS_PASSWORD,
    ssl=True,
    ssl_cert_reqs=None,
    ssl_check_hostname=False
)
