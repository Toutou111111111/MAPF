import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError()


class SecondOfMinute(TimeFeature):
    def __call__(self, index): return index.second / 59.0 - 0.5
class MinuteOfHour(TimeFeature):
    def __call__(self, index): return index.minute / 59.0 - 0.5
class HourOfDay(TimeFeature):
    def __call__(self, index): return index.hour / 23.0 - 0.5
class DayOfWeek(TimeFeature):
    def __call__(self, index): return index.dayofweek / 6.0 - 0.5
class DayOfMonth(TimeFeature):
    def __call__(self, index): return (index.day - 1) / 30.0 - 0.5
class DayOfYear(TimeFeature):
    def __call__(self, index): return (index.dayofyear - 1) / 365.0 - 0.5
class MonthOfYear(TimeFeature):
    def __call__(self, index): return (index.month - 1) / 11.0 - 0.5
class WeekOfYear(TimeFeature):
    def __call__(self, index): return (index.isocalendar().week.astype(int) - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str):
    freq_map = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
    }
    offset = to_offset(freq_str)
    for offset_type, feature_classes in freq_map.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]
    if str(freq_str).lower() in ['t', 'min', '15min']:
        return [MinuteOfHour(), HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]
    if str(freq_str).lower() in ['h', '1h']:
        return [HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]
    raise RuntimeError(f"Unsupported frequency {freq_str}")


def time_features(dates, freq='h'):
    dates = pd.DatetimeIndex(dates)
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
