from module.almazov_dataset_processing.data_analysis import *
from sklearn.linear_model import LinearRegression


class MeanImputationFilling(object):

    def fill(self, data_frame: pd.DataFrame, groups, target_columns: Dict[str, List]) -> [pd.DataFrame, int]:
        print('### === Start method : fill_missing_data.fill === ###')
        np.random.seed(1234)

        continuous = target_columns['continuous']
        categorical = target_columns['categorical']

        nans_summ = 0
        nans_summ += get_nans_count(data_frame, continuous)
        nans_summ += get_nans_count(data_frame, categorical)
        print('NaNs count before filling : {}'.format(nans_summ))

        result_frame = data_frame
        if continuous:
            result_frame[continuous] = result_frame.groupby(groups)[continuous].transform(lambda x: x.fillna(round(x.mean(), 2)))
            print(groups)
            names = []
            sizes = []
            for i, (name, group) in enumerate(result_frame.groupby(groups)):
                names.append('group_{}'.format(i))
                sizes.append(len(group))
            print(names, sizes)
        if categorical:
            result_frame[categorical] = result_frame.groupby(groups)[categorical].transform(lambda x: x.fillna(x.median()))

        print(result_frame.shape, data_frame.shape)
        assert (result_frame.shape == data_frame.shape)

        nans_summ = 0
        nans_summ += get_nans_count(data_frame, continuous)
        nans_summ += get_nans_count(data_frame, categorical)
        print('NaNs count after filling : {}'.format(nans_summ))

        print('### === End method : fill_missing_data.fill === ###')
        return result_frame, nans_summ


class LinearRegressionFilling(object):

    def __fill_one_column(self, data_frame, predictors, target):

        X = data_frame[predictors]
        Y = data_frame.loc[:, target]

        is_nans = pd.isna(Y)
        X_train = X[~is_nans].values
        y_train = Y[~is_nans].values
        X_test = X[is_nans].values

        alg = LinearRegression()
        alg.fit(X_train, y_train)
        y_pred = alg.predict(X=X_test)

        Y[is_nans] = y_pred

        return Y

    def fill(self, data_frame: pd.DataFrame, target_columns: Dict[str, List]) -> pd.DataFrame:

        result_frame = data_frame

        for target in target_columns:

            input_columns = data_frame.columns
            input_columns = input_columns.drop(
                ['patient_id', 'epizod_id', 'start_date', 'end_date', 'cod1', 'cod2', 'event_date', 'result',
                 'Department',
                 'gap', 'исход  -', 'date'])
            input_columns = input_columns.drop(data_frame)

            result_frame.loc[:, target] = self.__fill_one_column(data_frame, input_columns, target)
            print('{} passed'.format(target))

        return result_frame


def get_target_columns(continuous, categorical):
    if continuous and (type(continuous) is not list):
        continuous = [continuous]
    if categorical and (type(categorical) is not list):
        categorical = [categorical]

    return {'continuous': continuous, 'categorical': categorical}


def fill_by_groups(data_frame: pd.DataFrame, continuous: List[str], categorical: List[str]) -> pd.DataFrame:
    result_frame, _ = __condition_filling(data_frame)

    predictors = ['age', 'sex', 'sd', 'timedelta', 'condition']
    result_frame[predictors] = result_frame[predictors].astype(float)
    result_frame = result_frame.dropna(subset=predictors, how='any')

    filling_method = MeanImputationFilling()

    groups = __get_groups(result_frame)
    for group in groups:
        # print(group)
        result_frame, nans_summ = filling_method.fill(result_frame, group, get_target_columns(continuous, categorical))
        if not (nans_summ > 0):
            break

    return result_frame


def __condition_filling(data_frame: pd.DataFrame) -> [pd.DataFrame, int]:
    filling_method = MeanImputationFilling()
    result_frame, nans_count = filling_method.fill(data_frame, ['patient_id', 'epizod_id'], get_target_columns(None, 'condition'))
    return result_frame, nans_count


def __get_groups(data_frame: pd.DataFrame):
    age_bins = [24, 50, 60, 70, 80, 101]
    age_bins_df = pd.cut(data_frame['age'], age_bins)

    timedelta_bins = [i * 12 for i in range(-1, 62)]
    timedelta_bins_df = pd.cut(data_frame['timedelta'], timedelta_bins)

    group_1 = [age_bins_df, 'sex', 'sd', timedelta_bins_df, 'condition']
    group_2 = [age_bins_df, 'sex', 'sd', timedelta_bins_df]
    group_3 = ['sex', 'sd']

    age_bins = [24, 101]
    age_bins_df = pd.cut(data_frame['age'], age_bins)
    group_4 = [age_bins_df]

    groups = [group_1, group_2, group_3, group_4]

    return groups
