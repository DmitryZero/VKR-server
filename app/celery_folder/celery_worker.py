from celery import Celery

from app.config import settings, ssl_options

print(f'{settings.REDIS_URL}/0')
# celery_app = Celery("celery_worker", broker=f'{settings.REDIS_URL}/0', backend=f'{settings.REDIS_URL}/0')
celery_app = Celery("celery_worker", broker=f'{settings.REDIS_URL}/0', backend=None)

celery_app.conf.update(
    result_expires=3600,  # Время хранения результатов
    task_track_started=True
)

from app.celery_folder import tasks
