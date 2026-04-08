import sys
from loguru import logger
from argparse import ArgumentParser
from modules import (
    model,
    chunker,
    embedder
)


logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(
    'logs/main_logs.log',
    format = '{time} | {level} | {message}',
    level = 'DEBUG',
    rotation = '5 MB',
    retention = '5 days',
    compression = 'zip'
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
        logger.debug('Successful text parse')

    model.to(args.device)
    logger.info(f'Model run in {model.device}')

    logger.debug('Start run model.')
    embedding = embedder(
        text = raw_text,
        model = model,
        chunker = chunker
    )
    logger.debug('Successful model worked.')

    logger.debug('Start saving embedding in file.')
    with open(
        file = 'outputs/out.txt',
        mode = 'w'
        ) as file:

        file.write(str(embedding))
    logger.info('Successful write embedding in outputs/output.txt')


if __name__ == '__main__':
    main()