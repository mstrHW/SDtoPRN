from datetime import timedelta, datetime
from tqdm import tqdm

tqdm.pandas()
MY_DATE_FORMAT = '%Y%m%d%H%M'
NULL_DATE = '19990101' + '0' * 4
NULL_DATETIME = datetime.strptime(NULL_DATE, MY_DATE_FORMAT)


def timedelta_to_hours(time_delta: timedelta) -> float:
    hours = time_delta.total_seconds() / 3600.0
    hours = round(hours, 2)
    return hours


def datetime_from_string(date_string: str, date_format: str) -> datetime:
    value = datetime.strptime(date_string, date_format)
    return value


def datetime_to_string(date_time: datetime, date_format: str) -> str:
    return date_time.strftime(date_format)
