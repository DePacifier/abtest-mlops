import pandas as pd


def get_answered(df: pd.DataFrame, column1: str, column2: str) -> pd.DataFrame:
    return df[df[column1] == 1].append(df[df[column2] == 1])


def get_group(df: pd.DataFrame, from_column: str, classification: str) -> pd.DataFrame:
    return df[df[from_column] == classification]


def get_answered_results_combined(df: pd.DataFrame, classification_column: str, classications: list, column1: str, column2: str, new_column_name: str = 'response') -> pd.DataFrame:
    # check if really only 2 classifications provided
    assert(len(classications) == 2)
    # Separate the control and exposed first
    # Identifying the control group
    control_df = get_group(
        df, classification_column, classification=classications[0])
    # Identifying the exposed group
    exposed_df = get_group(
        df, classification_column, classification=classications[1])

    all_yes_df = control_df[control_df[column1] == 1].append(
        exposed_df[exposed_df[column1] == 1])
    all_yes_df = all_yes_df.drop([column1, column2], axis=1)
    all_yes_df[new_column_name] = 1
    # All users who answered no
    all_no_df = control_df[control_df[column2] == 1].append(
        exposed_df[exposed_df[column2] == 1])
    all_no_df = all_no_df.drop([column1, column2], axis=1)
    all_no_df[new_column_name] = 0
    # Final Combined dataframe will be
    combined_df = all_yes_df.append(all_no_df)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    return combined_df


def get_num_of_repetion(df: pd.DataFrame) -> int:
    # calculate unique values
    unique_values = df.auction_id.unique()
    # return the difference of the shape of the dataframe and unique values
    return df.shape[0] - len(unique_values)
