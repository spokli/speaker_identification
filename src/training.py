import os

import numpy as np
import pandas as pd

from src import config


def _load_features(first_n_files: int = None) -> dict[str, np.ndarray]:
    """load features from npy files

    Args:
        first_n_files (int, optional): number of files to load. For speedup during test runs. Defaults to None.

    Returns:
        dict[str, np.ndarray]: dictionary of feature with filename as key
    """
    filenames = os.listdir(config.PATH_DATA_FEATURES)
    filepaths = [
        os.path.join(config.PATH_DATA_FEATURES, filename)
        for filename in filenames
    ]
    if first_n_files is not None:
        first_n_files = int(min(first_n_files, len(filepaths)))
        filepaths = filepaths[:first_n_files]

    features = {}

    for filepath in filepaths:
        feature_arr = np.load(filepath)
        filename_noext = os.path.splitext(os.path.split(filepath)[-1])[0]
        features.update({filename_noext: feature_arr})

    return features


def _load_labels(first_n_files: int = None) -> pd.DataFrame:
    """load labels from csv file

    Args:
        first_n_files (int, optional): number of files to load. For speedup during test runs. Defaults to None.

    Returns:
        pd.DataFrame: dataframe of labels
    """
    filenames = os.listdir(config.PATH_DATA_LABELS)
    filepaths = [
        os.path.join(config.PATH_DATA_LABELS, filename)
        for filename in filenames
    ]
    if first_n_files is not None:
        first_n_files = int(min(first_n_files, len(filepaths)))
        filepaths = filepaths[:first_n_files]

    df_labels = None

    for filepath in filepaths:
        if df_labels is None:
            df_labels = pd.read_csv(filepath, sep=";")
        else:
            df_labels = df_labels.append(pd.read_csv(filepath, sep=";"))

    return df_labels


def _remove_duplicate_labels(df_labels: pd.DataFrame) -> pd.DataFrame:
    """remove duplicate labels from dataframe caused by an overwrite of labels

    Args:
        df_labels (pd.DataFrame): labels with duplicates

    Returns:
        pd.DataFrame: labels without duplicates
    """
    df = df_labels.copy()
    # sort by label_date and label_time, then drop duplicates
    df = df.sort_values(
        by=["filename", "label_date", "label_time"], ascending=True
    ).drop_duplicates(subset=["filename"], keep="last")

    return df
