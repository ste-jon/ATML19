import os

import pandas as pd
import gzip
from pathlib import Path
# from fetch_image import *
# from multiprocessing import Pool


def parse_data(fname):
  for l in open(fname):
    yield eval(l)


def parse_gzip(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF_gzip(path):
    i = 0
    df = {}
    for d in parse_gzip(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def getDF_csv(path, encoding, names):
    return pd.read_csv(path, encoding=encoding, names=names)


def getDF_json(path):
    return pd.read_json(path)


# def get_image_url(url):
#     html = fetch_html(url)
#     image_src = find_img_src(html)
#     return image_src


csv_names = ['asin', 'imageName', 'imageUrl', 'title', 'author', 'categoryId', 'category']
csv_df1 = getDF_csv('data/book30-listing-test.csv', encoding='utf_16_be', names=csv_names)
csv_df2 = getDF_csv('data/book30-listing-train.csv', encoding='utf_16_be', names=csv_names)
csv_df3 = getDF_csv('data/book32-listing.csv', encoding='iso-8859-1', names=csv_names)

csv_data = pd.concat([csv_df1, csv_df2, csv_df3])
csv_data.drop_duplicates('asin')
# csv_data['categories'] = csv_data.apply(lambda row: [row['category']], axis=1)
csv_data = csv_data.drop(columns=['imageName', 'title', 'author', 'categoryId'])
print(csv_data.shape)
csv_data = csv_data.dropna()
print(csv_data.shape)

# create file
os.makedirs('data/processed', exist_ok=True)
Path('data/processed/books_200000.json').touch()
csv_data.to_json('data/processed/books_200000.json', orient='records')


# json_list = list(parse_data('data/book_descriptions_50000.json'))
# json_data = pd.DataFrame.from_records(json_list, columns=['asin/ID', 'url', 'categories'])
# print(json_data.shape)
# print(json_data.iloc[0])
#
# print(json_data.loc[:10, 'url'])
#
# agents = 4
# chunksize = 3
# with Pool(processes=agents) as pool:
#     result = pool.map(get_image_url, json_data.loc[:10, 'url'], chunksize)
#
# print ('Result:  ' + str(result))