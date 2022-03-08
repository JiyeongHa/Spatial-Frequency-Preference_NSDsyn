import sys
import os
import numpy as np
import pandas as pd


def sub_number_to_string(sub_number):
    """ Return number (1,2,3,..) to "subj0x" form """
    return "subj%02d" % sub_number

def sort_a_df_column(df_vroinames):
    """ Input should be the whole column of a dataframe.
    Sort a column that contains either strings or numbers in a descending order"""

    roi_list = df_vroinames.unique().tolist()
    if all(isinstance(item, str) for item in roi_list):
        roi_list.sort(key=lambda x: int(x[1]))
    if all(isinstance(item, float) for item in roi_list):
        roi_list.sort(key=lambda x: int(x))

    return roi_list
def load_df(subj, df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
                          df_name='results_1D_model.csv'):
    sn = sub_number_to_string(subj)
    df_path = os.path.join(df_dir, sn + '_' + df_name)
    df = pd.read_csv(df_path)
    return df

def load_all_subj_df(subjects_to_run, df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
                                     df_name='results_1D_model.csv'):
    all_subj_df = []
    for sn in subjects_to_run:
        tmp_df = load_df(sn, df_dir=df_dir, df_name=df_name)
        all_subj_df.append(tmp_df)
    all_subj_df = pd.concat(all_subj_df, ignore_index=True)
    return all_subj_df

def create_empty_df(col_list=None):
    empty_df = pd.DataFrame(columns=col_list)
    return empty_df


def save_df_to_csv(df, output_dir, output_file_name, indexing=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_file_name)
    df.to_csv(output_path, index=indexing)
