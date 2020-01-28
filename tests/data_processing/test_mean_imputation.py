import numpy as np
import pandas as pd


df = pd.DataFrame(
    {
        'category': ['X', 'X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y'],
        'name': ['A','A', 'B','B','B','B', 'C','C','C'],
        'other_value': [10, np.nan, np.nan, 20, 30, 10, 30, np.nan, 30],
        'value': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        # 'value': [1, np.nan, np.nan, 2, 3, 1, 3, np.nan, 3],
    }
)

df2 = pd.DataFrame(
    {
        'category': ['X', 'X', 'X', 'X', 'X', 'X', 'Y'],
        'name': ['A', 'A', 'B', 'B', 'B', 'B', 'C'],
        'other_value': [10, np.nan, np.nan, 20, 30, 10, 30],
        'value': [1, np.nan, np.nan, 2, 3, 1, 3],
    }
)

groups = ['category', 'name']
columns = ['value']
new_frame = df.copy()
_new_frame = pd.concat([new_frame, df2], axis=0)
_new_frame[columns] = _new_frame.groupby(groups)[columns].transform(lambda x: x.fillna(x.mean()))
new_frame = _new_frame.iloc[:new_frame.shape[0], :]
# new_frame['value'] = new_frame.groupby(groups, sort=False)[columns].transform(lambda x: x.fillna(x.mean()))

print(new_frame['value'])
