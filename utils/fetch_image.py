import mimetypes
import multiprocessing
import os
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
    filename = os.path.join(save_path, name + '.jpeg')
    if skip_if_exists and os.path.isfile(filename):
        return
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    if img.format != 'JPEG': img = img.convert('RGB')
    img.save(filename, 'jpeg')


def do_fetch(d):
    count = 0
    for row in d.itertuples():
        asin = row.asin
        imageUrl = row.imageUrl
        try:
            fetch_image(imageUrl, 'data/coversraw/', asin)
        except Exception as e:
            print(asin + ' failed. ' + str(e))
            os.makedirs('data/processed', exist_ok=True)
            Path('data/processed/fails.csv').touch()
            with open('data/processed/fails.csv', 'a') as file:
                file.write(asin + '\n')
        count = count + 1
        if count % 100 == 0:
            print('Process ' + multiprocessing.current_process().name + ' ' + str(count) + '/' + str(d.shape[0]) + ' done.')

    return df


df = pd.read_json('data/processed/books_200000.json')
num_cores = multiprocessing.cpu_count() - 1  # leave one free to not freeze machine
num_partitions = num_cores  # number of partitions to split dataframe
df_split = np.array_split(df, num_partitions)
pool = multiprocessing.Pool(num_cores)
pool.map(do_fetch, df_split)
pool.close()
pool.join()
