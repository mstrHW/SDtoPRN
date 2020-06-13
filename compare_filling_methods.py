import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error

from module.almazov_dataset_processing.data_analysis import read_data, make_directory, path_join, get_nans_count, get_nans_percent, FINAL_DIR, EXPERIMENTS_DIR, os, get_range
from module.almazov_dataset_processing.filling_methods import MeanImputationFilling
import pickle


def linean_regression(train, test, predictor_columns, target_columns, experiment_path):
    train_X = train[predictor_columns]
    train_y = train[target_columns]

    test_X = test[predictor_columns]
    test_y = test[target_columns]

    for column in target_columns:
        train_target_column = train_y.loc[:, column]
        is_nan = pd.isna(train_target_column)

        not_nan_target_y = train_target_column[~is_nan].values
        not_nan_target_X = train_X[~is_nan].values

        alg = LinearRegression()
        alg.fit(not_nan_target_X, not_nan_target_y)

        test_target_column = test_y.loc[:, column]
        is_nan = pd.isna(test_target_column)

        test_not_nan_target_y = test_target_column[~is_nan].values
        test_not_nan_target_X = test_X[~is_nan].values

        pred_y = alg.predict(X=test_not_nan_target_X)

        target_column_results_path = path_join(experiment_path, column)
        make_directory(target_column_results_path)

        train_X_file = path_join(target_column_results_path, 'train_X')
        train_y_file = path_join(target_column_results_path, 'train_y')

        test_X_file = path_join(target_column_results_path, 'test_X')
        test_y_file = path_join(target_column_results_path, 'test_y')

        pred_y_file = path_join(target_column_results_path, 'pred_y')

        with open(train_X_file, 'wb') as file:
            pickle.dump(not_nan_target_X, file)

        with open(train_y_file, 'wb') as file:
            pickle.dump(not_nan_target_y, file)

        with open(test_X_file, 'wb') as file:
            pickle.dump(test_not_nan_target_X, file)

        with open(test_y_file, 'wb') as file:
            pickle.dump(test_not_nan_target_y, file)

        with open(pred_y_file, 'wb') as file:
            pickle.dump(pred_y, file)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_imputation(train, test, predictor_columns, target_columns, experiment_path):
    train_X = train[predictor_columns]
    train_y = train[target_columns]

    test_X = test[predictor_columns]
    test_y = test[target_columns]

    train_df = None
    for i, column in enumerate(target_columns):
        train_target_column = train_y.loc[:, column]
        is_nan = pd.isna(train_target_column)

        not_nan_target_y = train_target_column[~is_nan]
        not_nan_target_X = train_X[~is_nan]

        train_df = pd.concat([not_nan_target_X, not_nan_target_y], axis=1)

        # fill_values = get_fill_values(train_df, column, None)

        test_target_column = test_y.loc[:, column]
        is_nan = pd.isna(test_target_column)

        test_not_nan_target_y = test_target_column[~is_nan]
        test_not_nan_target_X = test_X[~is_nan]
        test_target_x_nans = test_not_nan_target_y.copy()
        test_target_x_nans.loc[:] = np.nan

        test_df = pd.concat([test_not_nan_target_X, test_target_x_nans], axis=1)

        df = pd.concat([train_df, test_df], axis=0)

        pred_df = fill_by_fill_values(df, column, None)
        _pred_df = pred_df.copy()
        pred_df = pred_df.iloc[train_df.shape[0]:, :]
        pred_y = pred_df[column]

        lower, upper, _ = get_range(pred_y)
        print('{} : {}..{}'.format(column, lower, upper))

        target_column_results_path = path_join(experiment_path, column)
        make_directory(target_column_results_path)

        train_X_file = path_join(target_column_results_path, 'train_X')
        train_y_file = path_join(target_column_results_path, 'train_y')

        test_X_file = path_join(target_column_results_path, 'test_X')
        test_y_file = path_join(target_column_results_path, 'test_y')

        pred_y_file = path_join(target_column_results_path, 'pred_y')

        with open(train_X_file, 'wb') as file:
            pickle.dump(not_nan_target_X.values, file)

        with open(train_y_file, 'wb') as file:
            pickle.dump(not_nan_target_y.values, file)

        with open(test_X_file, 'wb') as file:
            pickle.dump(test_not_nan_target_X.values, file)

        with open(test_y_file, 'wb') as file:
            pickle.dump(test_not_nan_target_y.values, file)

        with open(pred_y_file, 'wb') as file:
            pickle.dump(pred_y.values, file)

        if i == 0:
            new_y_df = _pred_df[column]
        else:
            new_y_df = pd.concat([new_y_df, _pred_df[column]], axis=1)

    new_df = pd.concat([train_X, test_X], axis=0)
    new_df = pd.concat([new_df, new_y_df], axis=1)

    return new_df


def get_fill_value(data_frame, groups, target_columns):
    print('### === Start method : fill_missing_data.fill === ###')
    np.random.seed(1234)

    # mask = lambda r: pd.isna(r) | pd.isnull(r)

    continuous = target_columns['continuous']
    categorial = target_columns['categorial']

    answer = None
    if continuous:
        # answer = data_frame.groupby(groups)[continuous].mean()
        answer = data_frame
    if categorial:
        answer = data_frame.groupby(groups)[categorial].median()

    return answer


def get_target_columns(continuous, categorial):
    if continuous and (type(continuous) is not list):
        continuous = [continuous]
    if categorial and (type(categorial) is not list):
        categorial = [categorial]

    return {'continuous': continuous, 'categorial': categorial}


def get_fill_values(data_frame, continuous, categorial):
    # condition = ['condition']
    # result_frame = data_frame.dropna(subset=condition)
    #
    # result_frame = fill(result_frame, ['patient_id', 'epizod_id'], get_target_columns(None, condition))
    fill_values = []

    input_columns = ['age', 'sex', 'sd', 'timedelta', 'condition']
    result_frame = data_frame.dropna(subset=input_columns)
    # result_frame = result_frame.dropna(subset=target_columns, how='all')

    age_bins = [24, 50, 60, 70, 80, 101]
    age_bins_df = pd.cut(result_frame['age'], age_bins)

    timedelta_bins = [i * 12 for i in range(-1, 62)]
    timedelta_bins_df = pd.cut(result_frame['timedelta'], timedelta_bins)

    groups = [age_bins_df, 'sex', 'sd', timedelta_bins_df, 'condition']
    fill_value = get_fill_value(result_frame, groups, get_target_columns(continuous, categorial))
    fill_values.append(fill_value)

    groups = [age_bins_df, 'sex', 'sd', timedelta_bins_df]
    fill_value = get_fill_value(result_frame, groups, get_target_columns(continuous, categorial))
    fill_values.append(fill_value)

    groups = ['sex', 'sd']
    fill_value = get_fill_value(result_frame, groups, get_target_columns(continuous, categorial))
    fill_values.append(fill_value)

    age_bins = [24, 101]
    age_bins_df = pd.cut(result_frame['age'], age_bins)
    groups = [age_bins_df]
    fill_value = get_fill_value(result_frame, groups, get_target_columns(continuous, categorial))
    fill_values.append(fill_value)

    return fill_values


def fill(data_frame, groups, target_columns):
    logging.info('### === Start method : fill_missing_data.fill === ###')
    np.random.seed(1234)

    # mask = lambda r: pd.isna(r) | pd.isnull(r)

    continuous = target_columns['continuous']
    categorial = target_columns['categorial']

    nans_summ = 0
    nans_summ += get_nans_count(data_frame, continuous)
    nans_summ += get_nans_count(data_frame, categorial)
    logging.info('NaNs count before filling : {}'.format(nans_summ))

    new_frame = data_frame.copy()
    if continuous:
        new_frame[continuous] = new_frame.groupby(groups)[continuous].transform(lambda x: x.fillna(x.mean()))
    if categorial:
        new_frame[categorial] = new_frame.groupby(groups)[categorial].transform(lambda x: x.fillna(fill_value))

    logging.info(new_frame.shape, data_frame.shape)
    assert(new_frame.shape == data_frame.shape)

    nans_summ = 0
    nans_summ += get_nans_count(data_frame, continuous)
    nans_summ += get_nans_count(data_frame, categorial)
    logging.info('NaNs count after filling : {}'.format(nans_summ))

    logging.info('### === End method : fill_missing_data.fill === ###')
    return new_frame


def fill_by_fill_values(data_frame, continuous, categorial):
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


def print_filling_results(filled_columns, lr_experiment_path):
    for column in filled_columns:
        column_path = path_join(lr_experiment_path, column)

        test_X_file = path_join(column_path, 'test_X')
        test_y_file = path_join(column_path, 'test_y')
        pred_y_file = path_join(column_path, 'pred_y')

        with open(test_y_file, 'rb') as f:
            test_y = pickle.load(f)

        with open(pred_y_file, 'rb') as f:
            pred_y = pickle.load(f)

        # max_error = (abs(test_y - pred_y)).max()
        # max_error2 = (abs(pred_y - test_y)).mean()

        mae = mean_absolute_error(test_y, pred_y)
        mape = mean_absolute_percentage_error(test_y, pred_y)
        print('{}: {} mae, {} mape'.format(column, round(mae, 4), round(mape, 4)))



def main():
    input_dir = FINAL_DIR
    data_frame = read_data(os.path.join(input_dir, 'dataset.csv'))

    experiment_path = path_join(EXPERIMENTS_DIR, 'compare_filling_methods')

    predictor_columns = ['sex', 'age', 'sd', 'condition', 'timedelta']
    filled_columns = ['Troponin', 'RBC', 'WBC', 'HGB', 'HCT', 'PLT', 'AST', 'ALT', 'Kreatinin',
                      'Glucose', 'Holesterin', 'pressure']

    print(data_frame.shape)
    data_frame = data_frame[predictor_columns + filled_columns].astype(float)
    probability = 0.8

    data_frame = data_frame.dropna(subset=predictor_columns)

    for column in data_frame.columns:
        lower, upper, _ = get_range(data_frame[column])
        nans_percent = round(get_nans_percent(data_frame, column), 4)
        print('{} : {}..{}, nans: {} %'.format(column, lower, upper, nans_percent * 100))

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data_frame, test_size=0.2, random_state=42)

    # lr_experiment_path = path_join(experiment_path, 'linear_regression')
    # linean_regression(train, test, predictor_columns, filled_columns, lr_experiment_path)
    # print_filling_results(filled_columns, lr_experiment_path)

    # mi_experiment_path = path_join(experiment_path, 'mean_imputation')
    # mean_imputation(train, test, predictor_columns, filled_columns, mi_experiment_path)
    # print_filling_results(filled_columns, mi_experiment_path)

    milr_experiment_path = path_join(experiment_path, 'mean_imputation_linear_regression')
    new_df = mean_imputation(train, test, predictor_columns, filled_columns, milr_experiment_path)
    print(new_df.columns)
    train, test = train_test_split(new_df, test_size=0.2, random_state=42)
    linean_regression(train, test, predictor_columns, filled_columns, milr_experiment_path)
    print_filling_results(filled_columns, milr_experiment_path)
    # for column in data_frame.columns:
    #     lower, upper, _ = get_range(data_frame[column])
    #     print('{} : {}..{}'.format(column, lower, upper))


if __name__ == '__main__':
    main()
