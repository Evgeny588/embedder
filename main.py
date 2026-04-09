import sys
from loguru import logger
from argparse import ArgumentParser
from modules import (
    model,
    chunker,
    embedder
)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--filename',
        type = str,
        default = 'none',
        help = '''Имя файла, которому нужен эмбеддинг, в формате example.txt'''
    )

    parser.add_argument(
        '--device',
        type = str,
        default = 'cpu',
        choices = ['cpu', 'cuda'],
        help = ''' Устройство, на котором будет работать модель ('cpu' или 'cuda')'''
    )

    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()

    if args.filename == 'none':
        logger.info('Необходимо ввести имя файла через аргумент --filename')
        sys.exit()
    else:
        with open(f'raw_text/{args.filename}') as file:
            raw_text = file.read()
        logger.debug(f'Successful text parse. Length of text = {len(raw_text)}')

    model.to(args.device)
    logger.info(f'Model run in {model.device}') # После отладки можно поменять уровень на DEBUG или удалить

    logger.debug('Start run model.')
    embedding = embedder(
        text = raw_text,
        model = model,
        chunker = chunker
    )

    logger.debug('Successful model worked. Start saving embbeding in file.')

    with open(
        file = 'outputs/out.txt', # Файл перезаписывается каждый раз, после вызова модели
        mode = 'w'
        ) as file:

        file.write(str(embedding))
    logger.info('Successful write embedding in outputs/output.txt')


if __name__ == '__main__':
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
    main()