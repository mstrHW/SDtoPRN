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

def prepare_by_timedelta(dates, values, need_add_dates, method='nearest'):

    result = np.array(values[0])
    _dates = dates.tolist()
    _values = values.tolist()

    for date in need_add_dates:
        prev_index = prev(_dates, date)
        next_index = next(_dates, date)

        prev_date = _dates[prev_index]
        prev_value = _values[prev_index]
        next_date = _dates[next_index]
        next_value = _values[next_index]

        delta = next_date - prev_date
        curr_delta = date - prev_date
        next_delta = next_date - date

        yi = 0
        if delta == 0.0:
            yi = next_value
        else:
            if method=='gradient':
                yi = get_by_grad(prev_value, next_value, delta, curr_delta)
            elif method=='nearest':
                if next_delta > curr_delta:
                    yi = next_value
                else:
                    yi = prev_value
        result = np.append(result, yi)

    return result


def prepare_dates(group_frame, target_columns, period, method='nearest') -> pd.DataFrame:

    period_td = period

    to_hours = lambda x: abs(x).total_seconds() / 3600.0
    dates = group_frame['event_date'].map(lambda x: datetime_from_string(x, MY_DATE_FORMAT))

    deltas_in_hours = dates.map(lambda x: to_hours(x - dates.iloc[0]))
    need_add_dates = get_needed_deltas(deltas_in_hours, period_td)

    deltas = np.append(deltas_in_hours.iloc[0], need_add_dates)
    new_dates = [dates.iloc[0] + timedelta(hours=hours) for hours in deltas]

    raws_count = len(new_dates)
    first_raw = group_frame.iloc[0]
    new_raws = [first_raw for i in range(raws_count)]

    new_df = pd.DataFrame(data=new_raws)
    new_df['event_date'] = new_dates
    new_df['event_date'] = new_df['event_date'].map(lambda x: datetime_to_string(x, MY_DATE_FORMAT))

    for column in target_columns:
        values = group_frame[column].values
        new_df[column] = prepare_by_timedelta(deltas_in_hours, values, need_add_dates, method)

    new_df['timedelta'] = deltas
    # new_df[new_df['event_timedelta'] > 0].loc[:, 'activ'] = 1
    # new_df[new_df['event_timedelta'] <= 0].loc[:, 'activ'] = 0

    return new_df


def get_needed_deltas(dates: pd.DataFrame, period: int):    # TODO: change function name
    start_date = dates.iloc[0]
    last_date = dates.iloc[-1]
    intermediate_count = int((last_date - start_date) / period)

    need_add_dates = [start_date + period * (i + 1) for i in range(intermediate_count)]

    return need_add_dates


def get_print_process_function(count: int):
    if count > 100:
        function = lambda i: i % int(count * 0.01) == 0
    else:
        function = lambda i: True

    return function