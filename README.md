# SDtoRNN
Автоматизированное построение моделей системной динамики. 
## main
### module/main/all_process.py (not refactored)
Процесс создания и настройки параметров модели sd для медицинских данных с различными режимами, а также построение графиков.
### module/main/data_preprocessing.py (almost refactored)
Процесс преобразования данных, включая очистку полей, соединение данных из разных файлов, обработка выбросов, заполнение пропусков, добавление периодичности и преобразование для rnn.
### module/main/simple_fd's_to_rnn.py (almost refactored)
Процесс создания и настройки параметров модели sd для данных простых моделей (teacup, predator-prey).

## Работа с данными

### module/almazov_dataset_processing/data_file_processing (refactored)
Содержит функции для работы с мед.данными центра Алмазова очистка данных, преобразование полей в вид, который может быть использован в моделях

### module/almazov_dataset_processing/convert_to_periodic_data.py (refactored)
Генерирует данные по заданному промежутку времени

### module/almazov_dataset_processing/data_analysis.py (refactored)
Помогает в выявлении проблемных мест в файле данных

### module/almazov_dataset_processing/filling_methods.py (refactored)
Содержит несколько методов заполнения пропусков (mean imputation (+многократный с разными группами), linear regression)

### module/almazov_dataset_processing/generate_data_for_rnn.py (not refactored)
Преобразование датасета для использования в rnn

### module/almazov_dataset_processing/merge_files.py (refactored)
Объединяет все файлы данных по заданным критериям

### module/almazov_dataset_processing/prepare_dataset.py (not refactored)
Добавляет рассчитываемые индикаторы, убирает выбросы

## Модели

### module/fd_model (refactored)
Содержит функции для работы с flow diagram (системная динамика)
#### fd_model.py - основные компоненты модели
#### vensim_fd_converter.py - модели, созданные в программе vensim преобразовываются во внутренние компоненты (fd_model.py)
#### fd_rnn_converter.py - преобразование модели в rnn (prn)

### module/print_results/stats.py (not refactored)
Содержит функции генерирования графиков для статьи

### module/pysd_simulation/pysd_simulation.py (not refactored)
Запускает различные режимы моделирования для моделей vensim (по исходным данным, с получение новых данных)

### module/vensim_models
Файлы моделей vensim (teacup, predator-prey, модель мед. индикаторов)

### module/nn_model.py (not refactored)
Модель сети в tensorflow

## Другое

### arch
Неиспользуемые на данный момент файлы с полезными функциями

### tests
Несколько тестов по работе с данными

### definitions.py
Функции и константы, необходимые во многих частях программы

### files_description.py (not refactored)
Описание файлов (должен переместиться в конфиг)
