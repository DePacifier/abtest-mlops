import numpy as np
import pandas as pd
import dvc.api
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Importing all data


def import_all_data_using_tagslist(path: str, repo: str, tags: list) -> dict:
    df_dict = {}
    for each in tags:
        data_url = dvc.api.get_url(path=path, repo=repo, rev=each)
        df_dict[each] = pd.read_csv(data_url)

    return df_dict

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


def split_date_to_numbers(df_dict: dict, date_column: str) -> dict:
    new_dict = df_dict.copy()
    for df in new_dict:
        new_dict[df][date_column] = pd.to_datetime(new_dict[df][date_column])
        new_dict[df]['year'] = new_dict[df][date_column].apply(
            lambda x: x.date().year)
        new_dict[df]['month'] = new_dict[df][date_column].apply(
            lambda x: x.date().month)
        new_dict[df]['day'] = new_dict[df][date_column].apply(
            lambda x: x.date().day)
        new_dict[df] = new_dict[df].drop(date_column, axis=1)

    return new_dict


def change_columns_to_numbers(df_dict: dict, columns: list) -> dict:
    lb = LabelEncoder()
    new_dict = df_dict.copy()
    for df in new_dict:
        df_columns = new_dict[df].columns.to_list()
        for change_col in columns:
            if(change_col in df_columns):
                new_dict[df][change_col] = lb.fit_transform(
                    new_dict[df][change_col])

    return new_dict


def get_train_validate_test_sets(df: pd.DataFrame, predicted_column: str, remove_columns: list, train_perc: float = 0.7, val_perc: float = 0.2, test_perc: float = 0.1) -> dict:
    data_dict = {}
    if(train_perc * 10 + val_perc * 10 + test_perc * 10 == 10):
        r_size = df.shape[0]
        train_columns = df.columns.to_list()
        train_columns.remove(predicted_column)

        for column in remove_columns:
            try:
                train_columns.remove(column)
            except:
                pass

        train_part = df.iloc[:int(r_size * (train_perc + val_perc)), :]
        test_part = df.iloc[int(r_size * (train_perc + val_perc)):, :]

        train_x, val_x, train_y, val_y = train_test_split(train_part.loc[:, train_columns],
                                                          train_part[predicted_column], test_size=val_perc)

        data_dict['train_x'] = train_x
        data_dict['train_y'] = train_y
        data_dict['val_x'] = val_x
        data_dict['val_y'] = val_y
        data_dict['test_x'] = test_part.loc[:, train_columns]
        data_dict['test_y'] = test_part.loc[:, predicted_column]

        return data_dict
    else:
        print("Invalid percentages used")
        return {}
