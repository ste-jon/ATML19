import os
from pathlib import Path
import shutil
import math
import pandas as pd

from utils.dataset_utils import category_to_folder, getDF_json

covers_extension = '.jpeg'
covers_src_dir = '../data/coversraw/'
covers_target_dir = '../data/covers/normalized/'
df_all = getDF_json('../data/processed/books_200000.json')
excludes = ['Gay & Lesbian', 'Education & Teaching']
normalize = True


def clear(root_dir):
    contents = os.listdir(root_dir)
    for content in contents:
        shutil.rmtree(os.path.join(root_dir, content))


def split(df, test_split, val_split, target_dir, normalize=False, excludes=None):
    # clear existing split
    clear(target_dir)
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
        copy(df_val, 'valid')
        copy(df_test, 'test')
        copy(df_train, 'train')
    else:
        num_test = math.floor(df.shape[0]*test_split)
        num_val = math.floor(df.shape[0]*val_split)
        print(df.shape[0])
        print(num_test, num_val)
        copy(df[:num_val], 'valid')
        copy(df[-num_test:], 'test')
        copy(df[num_val:-num_test], 'train')


def copy(df, set_name):
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
            print('copy ' + src_cover_fname + ' to ' + dest_cover_fname)
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


# df_all = df_all.sample(frac=1).reset_index(drop=True)
# sub_df = df_all[:10000]
split(df_all, .1, .1, covers_target_dir, normalize=True, excludes=excludes)
