import mimetypes
import multiprocessing
import os
from functools import partial
from pathlib import Path
from bs4 import BeautifulSoup
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import numpy as np


def fetch_html(url):
    contents = requests.get(url).content
    return contents


def find_img_src(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    img_tag = soup.find("img", {"id": "imgBlkFront"})
    return img_tag.get("src")


def fetch_image(url, save_path, name, skip_if_exists=True):
    fname = os.path.join(save_path, name)
    if skip_if_exists and (os.path.isfile(fname + '.jpe') or os.path.isfile(fname + '.jpeg')):
        return
    response = requests.get(url)
    content_type = response.headers['content-type']
    extension = mimetypes.guess_extension(content_type)
    img = Image.open(BytesIO(response.content))
    print(fname+extension)
    img.save(fname + extension)


def batch_fetch(d, save_dir, log_dir, verbose=False):
    count = 0
    for row in d.itertuples():
        asin = row.asin
        imageUrl = row.imageUrl
        try:
            fetch_image(imageUrl, save_dir, asin)
        except Exception as e:
            print(asin + ' failed. ' + str(e))
            os.makedirs(dir, exist_ok=True)
            fname = os.path.join(log_dir, 'fails.csv')
            print(fname)
            Path(fname).touch()
            with open(fname, 'a') as file:
                file.write(asin + '\n')
        count = count + 1
        if verbose and count % 100 == 0:
            print('Process ' + multiprocessing.current_process().name + ' ' + str(count) + '/' + str(d.shape[0]) + ' done.')
    return d

def download_covers(json_filename, covers_save_dir, log_dir, verbose=False):
    df = pd.read_json(json_filename)
    num_cores = multiprocessing.cpu_count() - 1  # leave one free to not freeze machine
    num_partitions = num_cores  # number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    fetch_partial = partial(batch_fetch, save_dir=covers_save_dir, log_dir=log_dir, verbose=verbose)
    pool = multiprocessing.Pool(num_cores)
    pool.map(fetch_partial, df_split)
    pool.close()
    pool.join()
    print('Done')