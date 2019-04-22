from arch.Preprocessing import *
from arch.Loader import Loader


def gradient_test():
    prev_value = 0
    next_value = 20
    delta = 500
    need_delta = 50
    yi = get_by_grad(prev_value, next_value, delta, need_delta)
    expected = 2.0
    assert(yi == expected)

def prepare_by_timedelta_test():
    dates = [1, 2, 13, 15]
    values = [1, 2, 13, 15]
    max_delta = 5
    yi = prepare_by_timedelta(dates, values, max_delta)
    expected = [1., 6., 11.]
    for i in range(len(yi)):
        assert(yi[i]==expected[i])


if __name__ == '__main__':
    loader = Loader()
    data_path = loader.data_path
    new_holesterin_file = data_path + 'prepared_holesterin.csv'
    new_labresult_file = data_path + 'prepared_labresult.csv'
    # prepare_labresult_file()
    # dataset = loader.read_data(new_labresult_file)

    gradient_test()
    prepare_by_timedelta_test()
