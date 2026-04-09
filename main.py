import sys
from loguru import logger
from argparse import ArgumentParser, Namespace
from pathlib import Path
from modules import model, chunker, embedder


def parse_args() -> Namespace:
    """Парсит аргументы командной строки: имя файла и устройство для модели."""
    parser = ArgumentParser()
    parser.add_argument(
        '--filename',
        type=str,
        default='none',
        help='Имя файла для эмбеддинга в формате example.txt'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Устройство для запуска модели: cpu или cuda'
    )
    return parser.parse_args()


def main() -> None:
    """Основной пайплайн: загрузка текста → инференс модели → сохранение эмбеддинга."""
    args = parse_args()

    if args.filename == 'none':
        logger.error('Необходимо указать имя файла через аргумент --filename')
        sys.exit(1)

    try:
        file_path = Path('raw_text') / args.filename
        raw_text = file_path.read_text(encoding='utf-8')
        logger.debug(f'Successful text parse. Length of text = {len(raw_text)}')
    except FileNotFoundError:
        logger.error(f'Файл не найден: {file_path}')
        sys.exit(1)
    except PermissionError:
        logger.error(f'Нет прав на чтение файла: {file_path}')
        sys.exit(1)
    except Exception as e:
        logger.exception(f'Неожиданная ошибка при чтении файла: {e}')
        sys.exit(1)

    model.to(args.device)
    logger.info(f'Model run on {model.device}')

    logger.debug('Start run model.')
    embedding = embedder(text=raw_text, model=model, chunker=chunker)
    logger.debug('Successful model worked. Start saving embedding in file.')

    try:
        output_path = Path('outputs/out.txt')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(embedding), encoding='utf-8')
        logger.info(f'Successful write embedding to {output_path}')
    except PermissionError:
        logger.error(f'Нет прав на запись в файл: {output_path}')
        sys.exit(1)
    except OSError as e:
        logger.error(f'Ошибка при записи файла: {e}')
        sys.exit(1)


if __name__ == '__main__':
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        'logs/logs.log',
        format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {file}:{line} | {message}',
        level='DEBUG',
        rotation='5 MB',
        retention='5 days',
        compression='zip'
    )
    main()