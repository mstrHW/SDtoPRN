import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from module.data_loader import loader
from data_processing.filling_method import FillingMethod
from data_processing.analyze_data import *
from main.files_columns import files_description


def linean_regression(df, input_columns, target_columns):
    _df = df

    for column in target_columns:
        input_columns = df.columns
        input_columns = input_columns.drop(
            ['patient_id', 'epizod_id', 'start_date', 'end_date', 'cod1', 'cod2', 'event_date', 'result', 'Department',
             'gap', 'исход  -', 'date'])
        input_columns = input_columns.drop(column)

        X = df[input_columns]

        _column = df.loc[:, column]

        mask = lambda r: pd.isna(r)
        is_nans = mask(_column)
        X_train = X[~is_nans].values
        y_train = _column[~is_nans].values
        X_test = X[is_nans].values

        alg = LinearRegression()
        alg.fit(X_train, y_train)

        y_pred = alg.predict(X=X_test)

        # _column = _column.astype(float).values
        # _column = _column.apply(lambda r: alg.predict(X=r) if mask(r) else r)
        _column[is_nans] = y_pred
        # for i in range(_column.shape[0]):
        #     try:
        #         _column[i] = alg.predict(X=_column[i]) if mask(_column[i]) else _column[i]
        #     except:
        #         print(_column[i])
        _df.loc[:, column] = _column

        print('{} passed'.format(column))

    return _df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def lr_filling(dataset, input_columns, target_column):
    k = 10
    kf = KFold(n_splits=k)
    regrs = [LinearRegression]
    my_algorithms = [FillingMethod]
    # regression_algorithm = LogisticRegression()
    error_function = mean_absolute_error
    maes_regr = np.array([])
    maes_grouped = np.array([])
    mapes_regr = np.array([])
    mapes_grouped = np.array([])
    _maes_regr = np.array([])
    _mapes_regr = np.array([])

    input_columns = dataset.columns
    input_columns = input_columns.drop(
        ['patient_id', 'epizod_id', 'start_date', 'end_date', 'cod1', 'cod2', 'event_date', 'result', 'Department',
         'gap', 'исход  -', 'date'])
    input_columns = input_columns.drop(target_column)
    current_data = dataset[input_columns]
    current_target = dataset[target_column]
    _data = current_data
    _target = current_target
    _data = _data.astype(float).values
    _target = _target.astype(float).values

    i = 0
    for train_index, test_index in kf.split(_data):
    # for i in range(9, k):
        i += 1

        # print('{} : {}'.format(_data.shape, dataset.shape))
        np.save('tmp/train_{}{}.npy'.format(target_column, i), train_index)
        np.save('tmp/test_{}{}.npy'.format(target_column, i), test_index)
        # train_index = np.load('tmp/train_{}{}.npy'.format(target_column, i))
        # test_index = np.load('tmp/train_{}{}.npy'.format(target_column, i))
        X_train, X_test = _data[train_index], _data[test_index]
        y_train, y_test = _target[train_index], _target[test_index]

        # for regr in regrs:
        #     alg = regr()
        #     alg.fit(X_train, y_train)
        #     y_pred = alg.predict(X=X_test)
        #     mae = error_function(y_test, y_pred)
        #     mape = mean_absolute_percentage_error(y_test, y_pred)
        #     maes_regr = np.append(maes_regr, mae)
        #     mapes_regr = np.append(mapes_regr, mape)
        #
        # _result_frame = []
        # for _alg in my_algorithms:
        #     alg = _alg()
        #     _dataset = dataset.loc[:, dataset.columns]
        #     y_test = _dataset[target_column].values[test_index]
        #     _dataset[target_column].values[test_index] = np.nan
        #
        #     age_bins = [24, 50, 60, 70, 80, 101]
        #     age_bins_df = pd.cut(_dataset['age'], age_bins)
        #
        #     timedelta_bins = [i*12 for i in range(-1, 62)]
        #     timedelta_bins_df = pd.cut(_dataset['timedelta'], timedelta_bins)
        #
        #     groups = [age_bins_df, 'sex', 'sd', timedelta_bins_df, 'condition']
        #
        #     result_frame = alg.fill(_dataset, groups, get_target_columns(target_column, None))
        #     y_pred = result_frame[target_column].values[test_index]
        #
        #     if np.isnan(y_pred).any():
        #         groups = [age_bins_df, 'sex', 'sd', timedelta_bins_df]
        #         result_frame = alg.fill(result_frame, groups, get_target_columns(target_column, None))
        #         y_pred = result_frame[target_column].values[test_index]
        #
        #     if np.isnan(y_pred).any():
        #         groups = ['sex', 'sd']
        #         result_frame = alg.fill(result_frame, groups, get_target_columns(target_column, None))
        #         y_pred = result_frame[target_column].values[test_index]
        #     # print('y shape : {}'.format(y_pred.shape[0]))
        #     if np.isnan(y_pred).any():
        #         # print('here1')
        #         # print(np.count_nonzero(~np.isnan(y_pred)))
        #         age_bins = [24, 101]
        #         age_bins_df = pd.cut(result_frame['age'], age_bins)
        #         groups = [age_bins_df]
        #         result_frame = alg.fill(result_frame, groups, get_target_columns(target_column, None))
        #         y_pred = result_frame[target_column].values[test_index]
        #
        #
        #     np.save('tmp/y_{}{}.npy'.format(target_column, i), y_pred)
        #     assert (result_frame.shape == _dataset.shape)
        #
        #     mae = error_function(y_test, y_pred)
        #     mape = mean_absolute_percentage_error(y_test, y_pred)
        #     maes_grouped = np.append(maes_grouped, mae)
        #     mapes_grouped = np.append(mapes_grouped, mape)
        #     _result_frame = result_frame.loc[:, result_frame.columns]

        for regr in regrs:
            alg = regr()
            alg.fit(X_train, y_train)
            y_pred = alg.predict(X=X_test)
            _mae = error_function(y_test, y_pred)
            _mape = mean_absolute_percentage_error(y_test, y_pred)
            _maes_regr = np.append(maes_regr, _mae)
            _mapes_regr = np.append(mapes_regr, _mape)

        print('{} : {} step passed'.format(target_column, i))

    return maes_regr, mapes_regr, maes_grouped, mapes_grouped, _maes_regr, _mapes_regr

def compare_filling_methods(new_dataset, input_columns, target_columns):
    for target_column in target_columns:
        dataset = new_dataset.dropna(subset=[target_column])
        # current_data = new_dataset[input_columns]
        # current_target = new_dataset[target_column]

        print('{} : {}'.format(target_column, dataset[target_column].count()))
        mses = np.array(lr_filling(dataset, input_columns, target_column))

        np.save('tmp/{}_errors.npy'.format(target_column), mses)
        # np.save('tmp/{}_my_algorithm.npy'.format(target_column), mses2)

        # print('{}_lin_regression'.format(target_column, mses.mean()))
        # print('{}_my_algorithm'.format(target_column, mses2.mean()))

def get_results(target_columns):
    for target_column in target_columns:
        mses = np.load('tmp/{}_errors.npy'.format(target_column))
        # print('{} : {}'.format(target_column, [error.mean() for error in mses]))
        print('{:2.2f}\t{:2.2f}\t{:2.2f}'.format(round(mses[0].mean(), 2), round(mses[2].mean(), 2), round(mses[4].mean(), 2)))

def get_print_process_function(count):
    if count > 100:
        function = lambda i: i % int(count * 0.01) == 0
    else:
        function = lambda i: True

    return function

def fill(data_frame, groups, target_columns):
    print('### === Start method : fill_missing_data.fill === ###')
    np.random.seed(1234)

    # mask = lambda r: pd.isna(r) | pd.isnull(r)

    continuous = target_columns['continuous']
    categorial = target_columns['categorial']

    nans_summ = 0
    nans_summ += get_nans_count(data_frame, continuous)
    nans_summ += get_nans_count(data_frame, categorial)
    print('NaNs count before filling : {}'.format(nans_summ))

    new_frame = data_frame
    if continuous:
        new_frame[continuous] = new_frame.groupby(groups)[continuous].transform(lambda x: x.fillna(x.mean()))
    if categorial:
        new_frame[categorial] = new_frame.groupby(groups)[categorial].transform(lambda x: x.fillna(x.median()))

    print(new_frame.shape, data_frame.shape)
    assert(new_frame.shape == data_frame.shape)

    nans_summ = 0
    nans_summ += get_nans_count(data_frame, continuous)
    nans_summ += get_nans_count(data_frame, categorial)
    print('NaNs count after filling : {}'.format(nans_summ))

    print('### === End method : fill_missing_data.fill === ###')
    return new_frame

def get_target_columns(continuous, categorial):
    if continuous and (type(continuous) is not list):
        continuous = [continuous]
    if categorial and (type(categorial) is not list):
        categorial = [categorial]

    return {'continuous': continuous, 'categorial': categorial}


def fill_by_groups(data_frame, continuous, categorial):
    # condition = ['condition']
    # result_frame = data_frame.dropna(subset=condition)
    #
    # result_frame = fill(result_frame, ['patient_id', 'epizod_id'], get_target_columns(None, condition))

    input_columns = ['age', 'sex', 'sd', 'timedelta', 'condition']
    result_frame = data_frame.dropna(subset=input_columns)
    # result_frame = result_frame.dropna(subset=target_columns, how='all')

    age_bins = [24, 50, 60, 70, 80, 101]
    age_bins_df = pd.cut(result_frame['age'], age_bins)

    timedelta_bins = [i * 12 for i in range(-1, 62)]
    timedelta_bins_df = pd.cut(result_frame['timedelta'], timedelta_bins)

    groups = [age_bins_df, 'sex', 'sd', timedelta_bins_df, 'condition']
    result_frame = fill(result_frame, groups, get_target_columns(continuous, categorial))

    groups = [age_bins_df, 'sex', 'sd', timedelta_bins_df]
    result_frame = fill(result_frame, groups, get_target_columns(continuous, categorial))

    groups = ['sex', 'sd']
    result_frame = fill(result_frame, groups, get_target_columns(continuous, categorial))

    age_bins = [24, 101]
    age_bins_df = pd.cut(result_frame['age'], age_bins)
    groups = [age_bins_df]
    result_frame = fill(result_frame, groups, get_target_columns(continuous, categorial))

    return result_frame

def main():
    data_path = 'C:/Datasets/RW_3/RW_3/generated/'
    file = data_path + 'dataset2_Td_Th.csv'
    # file = data_path + 'full_dataset_PC_PD_PT2.csv'
    new_dataset = loader.read_data(file)
    print(new_dataset.shape)

    input_columns = ['age', 'sex', 'sd', 'timedelta', 'condition']
    lab_desc = files_description['labresult_file']
    hol_desc = files_description['holesterin_file']
    prd_desc = files_description['predst_file']
    eho_desc = files_description['eho_file']
    continuous = lab_desc['float_columns'] + hol_desc['float_columns'][1:] + prd_desc['float_columns']\
                     + eho_desc['float_columns']
    categorial = lab_desc['categorial_columns'] + hol_desc['categorial_columns'] + prd_desc['categorial_columns']\
                     + eho_desc['categorial_columns']
    # print(get_unique_symbols(new_dataset, 'condition'))
    # target_columns = ['RBC', 'Troponin', 'RBC', 'WBC', 'HGB', 'HCT', 'pressure']

    # input_columns = ['sex', 'timedelta', 'condition']
    # target_columns = files_description['cpidcod_file']['float_columns']

    #? ['AST', 'ALT',
    #['PLT', 'Kreatinin', 'Glucose', 'Holesterin', ]
    # print(new_dataset.shape[0])
    target_columns = continuous + categorial
    new_dataset[input_columns] = new_dataset[input_columns].astype(float)
    new_dataset[target_columns] = new_dataset[target_columns].astype(float)

    new_dataset = new_dataset.dropna(subset=input_columns)
    new_dataset = new_dataset.dropna(subset=target_columns, how='all')

    # new_dataset = new_dataset.dropna(subset=categorial)
    # lower, upper, _ = get_range(new_dataset[categorial])
    # print('{} : {}..{}'.format(categorial, lower, upper))


    # new_dataset[categorial] = (new_dataset[categorial]-1)
    # new_dataset[categorial] = new_dataset[categorial] / new_dataset[categorial].max()
    # print(new_dataset.shape[0])
    # print('Input columns')
    # for column in input_columns:
    #     lower, upper, _ = get_range(new_dataset[column])
    #     print('{} : {}..{}'.format(column, lower, upper))


    new_dataset = fill_by_groups(new_dataset, continuous, categorial)
    loader.write_data(new_dataset, data_path + 'dataset2_FbG.csv')

    # print('### === Output columns === ###')
    # for column in target_columns:
    #     lower, upper, _ = get_range(new_dataset[column])
    #     print('{} : {}..{}'.format(column, lower, upper))
    # print('### === Nans count === ###')
    # print(get_nans_percent(new_dataset, target_columns))
    # print(new_dataset.shape)
    # loader.write_data(new_dataset, data_path + 'dataset_FbGr.csv')

    # new_dataset = linean_regression(new_dataset, input_columns, target_columns) # TODO

    # print('Output columns')
    # for column in target_columns:
    #     lower, upper, _ = get_range(new_dataset[column])
    #     print('{} : {}..{}'.format(column, lower, upper))
    # print(get_range(new_dataset))
    # print(get_nans_percent(new_dataset, target_columns))
    # print(new_dataset.shape)

    # compare_filling_methods(new_dataset, input_columns, target_columns)
    # get_results(target_columns)

    # loader.write_data(new_dataset, data_path + 'dataset_FbG+Lr.csv')  # TODO

if __name__ == '__main__':
    ing = 0
    main()
    pass