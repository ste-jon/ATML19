import os
from pathlib import Path
import shutil
import math

from utils.dataset_utils import category_to_folder, getDF_json

covers_extension = '.jpeg'
covers_src_dir = '../data/coversraw/'
covers_target_dir = '../data/covers/sample/'
df_all = getDF_json('../data/processed/books_200000.json')


def clear(root_dir):
    contents = os.listdir(root_dir)
    for content in contents:
        shutil.rmtree(os.path.join(root_dir, content))


def split(df, test_split, val_split, target_dir):
    # clear existing split
    clear(target_dir)
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
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


# split(df_all, .1, .1, covers_target_dir)

# sample
df_all = df_all.sample(frac=1).reset_index(drop=True)
sub_df = df_all[:10000]
split(sub_df, .1, .1, covers_target_dir)
