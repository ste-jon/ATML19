import os
from pathlib import Path
import shutil
import math
import pandas as pd

from utils.dataset_utils import category_to_folder, getDF_json

def clear(root_dir):
    contents = os.listdir(root_dir)
    for content in contents:
        shutil.rmtree(os.path.join(root_dir, content))


def split(df, test_split, val_split, src_dir, target_dir, covers_extension, normalize=False, excludes=None, clear=False):
    # clear existing split
    if clear: clear(target_dir)
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    if excludes:
        df = exclude(df, excludes)
    if normalize:
        counts = count_categories(df)
        sorted_by_count = sorted(counts, key=lambda tup: tup[1])
        lowest_count = sorted_by_count[0]
        print('Lowest count is "' + lowest_count[0] + '" with ' + str(lowest_count[1]))
        df = select_num_of_cat(df, lowest_count[1])
        num_test = math.floor(lowest_count[1] * test_split)
        num_val = math.floor(lowest_count[1] * val_split)
        # first extract num_test from df
        df_test = select_num_of_cat(df, num_test)
        # 'subtract' df_test from dr
        df = pd.concat([df, df_test]).drop_duplicates(keep=False)
        df_val = select_num_of_cat(df, num_val)
        df_train = pd.concat([df, df_val]).drop_duplicates(keep=False)
        copy(df_val, 'valid', src_dir, target_dir, covers_extension)
        copy(df_test, 'test', src_dir, target_dir, covers_extension)
        copy(df_train, 'train', src_dir, target_dir, covers_extension)
    else:
        num_test = math.floor(df.shape[0]*test_split)
        num_val = math.floor(df.shape[0]*val_split)
        print(df.shape[0])
        print(num_test, num_val)
        copy(df[:num_val], 'valid', src_dir, target_dir, covers_extension)
        copy(df[-num_test:], 'test', src_dir, target_dir, covers_extension)
        copy(df[num_val:-num_test], 'train', src_dir, target_dir, covers_extension)


def copy(df, set_name, covers_src_dir, covers_target_dir, covers_extension):
    for row in df.itertuples():
        cat_name = row.category
        asin = row.asin
        src_cover_fname = os.path.join(covers_src_dir, asin + covers_extension)
        cover_file = Path(src_cover_fname)
        if cover_file.is_file():
            cat_folder_name = category_to_folder(cat_name)
            target_dir = os.path.join(covers_target_dir, set_name, cat_folder_name)
            dest_cover_fname = os.path.join(target_dir, asin + covers_extension)
            os.makedirs(os.path.dirname(dest_cover_fname), exist_ok=True)
            # print('copy ' + src_cover_fname + ' to ' + dest_cover_fname)
            shutil.copyfile(src_cover_fname, dest_cover_fname)
        else: raise Exception('file not found')


def count_categories(df):
    counts = []
    for cat in df['category'].unique():
        count = len(df[df['category'] == cat])
        counts.append((cat, count))
    return counts


def exclude(df, excludes):
    filtered_df = df
    for ex in excludes:
        print('Dropping ' + ex)
        filtered_df = filtered_df[filtered_df.category != ex]
        print(filtered_df.shape[0])
    return filtered_df


def select_num_of_cat(df, num):
    slices = []
    for cat in df['category'].unique():
        tmp_df = df
        tmp_df = tmp_df[tmp_df['category'] == cat]
        tmp_df = tmp_df[:num]
        slices.append(tmp_df)
    return pd.concat(slices)


def create_image_folders(ds_json, covers_src_dir, covers_target_dir, test_split=0.1, val_split=0.1, normalize=False, excludes=None):
    df = getDF_json(ds_json)
    df = df.sample(frac=1).reset_index(drop=True)
    split(df, test_split, val_split, covers_src_dir, covers_target_dir, '.jpeg', normalize=normalize, excludes=excludes)
    print('Done')
