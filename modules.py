from sentence_transformers import SentenceTransformer
from chonkie import SentenceChunker
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

import torch
import os
import sys

logger.remove()
logger.add(
    sys.stderr,
    format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"  # Чтобы в консоль выводились только логи уровня INFO
)
logger.add(
    'logs/logs.log',
    format = '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {file}:{line} | {message}',
    level = 'DEBUG', # В файл пишутся все логи, начиная с уровня DEBUG
    rotation = '5 MB',
    retention = '5 days',
    compression = 'zip'
)

load_dotenv()

flag_cache_model = any(Path('model_cache/').iterdir())
logger.debug(f'Cache model is {flag_cache_model}')

model = SentenceTransformer(
    model_name_or_path = os.getenv('CHECKPOINT'),
    cache_folder = 'model_cache/',
    local_files_only = flag_cache_model
)

chunker = SentenceChunker(
    tokenizer = model.tokenizer,
    chunk_size = 450,
    chunk_overlap = 50
)


def embedder(
      text: str,
      model: SentenceTransformer,
      chunker: SentenceChunker
      ) -> list[float]:
    """
    Преобразует текст в векторное представление (эмбеддинг).
    
    Текст разбивается на чанки, каждый кодируется моделью, 
    затем применяется гибридное пулинг-агрегирование: (mean + max) / 2.
    
    Args:
        text: Исходный текст для эмбеддинга.
        model: Загруженная модель SentenceTransformer.
        chunker: Экземпляр SentenceChunker для токенизации.
    
    Returns:
        list[float]: Финальный вектор-эмбеддинг в виде списка чисел.
    """

    chunks = chunker.chunk(text)
    logger.debug(f'Num chunks = {len(chunks)}')
    chunk_texts = ['passage: ' + c.text for c in chunks]

    with torch.inference_mode():
      embeddings = model.encode(chunk_texts, convert_to_tensor=True)
    logger.debug(f'Embeddings shape = {embeddings.shape}')

    mean_emb = torch.mean(embeddings, dim=0)

    max_emb, _ = torch.max(embeddings, dim=0)

    final_vector = (mean_emb + max_emb) / 2
    return final_vector.cpu().tolist()