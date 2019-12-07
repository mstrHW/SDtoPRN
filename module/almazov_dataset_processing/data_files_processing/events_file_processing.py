import pandas as pd

from definitions import *
from files_description import events_description
from module.almazov_dataset_processing.data_files_processing.columns_processing import prepare_columns, delete_nan_ids, time_to_my_format
from module.almazov_dataset_processing.data_analysis import get_unique_symbols
from module.data_loader import read_data, write_to_csv


def __read_raw_events_file(file_name: str, id_columns: List[str], DEMO: bool = False) -> pd.DataFrame:
    data_frame = read_data(file_name, encoding='UTF-8', delimiter='\t')

    if DEMO:
        sample = data_frame.head(1000)
        data_frame = sample
        write_to_csv(sample, path_join(BASE_FILES, 'events_sample.csv'))

    data_frame = delete_nan_ids(data_frame, id_columns)
    data_frame['time'] = data_frame['time'].apply(time_to_my_format)
    data_frame['data'] += data_frame['time']
    data_frame = data_frame.drop('time', axis=1)
    return data_frame


def __standardize_events(data_frame: pd.DataFrame) -> pd.DataFrame:
    print('before standardize_events {}'.format(data_frame.shape))

    uniq = get_unique_symbols(data_frame, 'activ')
    # uniq.__delitem__(np.nan)
    uniq.__delitem__('')

    stopwords = ['регистрация', 'выписка', 'отмена', 'осмотр', 'консультация', 'уровня', 'рентгенография', 'анализ',
                 'томография', 'документ', 'тест', 'мониторирование', 'узи_', 'мрт_', 'экспресс_', 'определение',
                 'исследование', 'взятие', 'кт_', 'расчет', 'перевод', 'поступление_', 'данных_', 'уведомление',
                 'отменено', 'экспр.лаб.', 'мс_кт', 'группа_крови_и_резус-фактор', 'перенесено', 'скрининг',
                 'абонемент',
                 'выдача', 'выявление', 'эхокардиография', 'аудиометрия', 'оценка', 'тонометрия', 'расшифровка',
                 'описание',
                 'графи', 'наблюдение', 'отправка', 'направление', 'процедуры', 'курация', 'мониторинг', 'диагностика',
                 'метрия',
                 'физкультура', 'уход']
    stopwords_not_in = lambda symbol: not max([(word in symbol.lower()) for word in stopwords])
    # gowords = ['гимнастика', 'занятие', 'массаж', 'процедуры']
    gowords = ['операция', 'пластика', 'наложение', 'сшивание', 'шунтирование',
               'облучение', 'стентирование']
    gowords_in = lambda symbol: max([(word in symbol.lower()) for word in gowords])

    uniq = [symbol for symbol in uniq if stopwords_not_in(symbol) and gowords_in(symbol)]

    data_frame = data_frame[data_frame['activ'].isin(uniq)]
    data_frame.loc[:, 'activ'] = 1
    print('after standardize_events {}'.format(data_frame.shape))

    return data_frame


def main(input_file: str, output_file: str) -> None:
    logging.info('Start processing events file')
    data = __read_raw_events_file(input_file, events_description['id_columns'])
    data = __standardize_events(data)
    data = prepare_columns(data, events_description, '%Y%m%d%H%M')

    write_to_csv(data, output_file)
    logging.info('Events was processed')


if __name__ == '__main__':
    __DEMO = False
    output_file_name = 'events.csv'
    if __DEMO:
        output_file_name = 'events_sample.csv'

    main(path_join(BASE_FILES, FILE_NAMES[3]), path_join(CLEARED_DIR, output_file_name))
