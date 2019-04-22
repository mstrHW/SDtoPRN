import numpy as np

def next(list, value):
    for i in list:
        if i >= value:
            return list.index(i)
    return -1

def prev(list, value):
    prev_value = -1
    for i in list:
        if i <= value:
            prev_value = list.index(i)
        else:
            break
    return prev_value

def get_by_grad(prev_value, next_value, next_delta, need_delta):
    dy = (next_value - prev_value) / next_delta
    yi = dy * need_delta + prev_value
    return yi

def prepare_by_timedelta(dates, values, need_add_dates):

    result = np.array(values[0])

    for date in need_add_dates:
        prev_index = prev(dates, date)
        next_index = next(dates, date)

        prev_date = dates[prev_index]
        prev_value = values[prev_index]
        next_date = dates[next_index]
        next_value = values[next_index]

        next_delta = next_date - prev_date
        curr_delta = date - prev_date

        if next_delta == 0.0:
            yi = next_value
        else:
            yi = get_by_grad(prev_value, next_value, next_delta, curr_delta)
        result = np.append(result, yi)

    return result

