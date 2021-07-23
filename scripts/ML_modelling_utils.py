import numpy as np
import pandas as pd
# Selecting only browsers with more than 100 rows or users


def get_index_based_on_size(size: np.array, value: int) -> list:
    size = size.tolist()
    index_list = []
    for index in range(len(size)):
        if(size[index] >= value):
            index_list.append(index)

    return index_list


def get_df_of_each_group(grouped_df: pd.DataFrame, selected_index: list) -> list:
    size_series = grouped_df.size()
    df_list = []
    for index in selected_index:
        df = grouped_df.get_group(size_series.index[index])
        df_list.append(df.reset_index().drop('index', axis=1))

    return df_list
