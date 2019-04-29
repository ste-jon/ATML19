import os
import pandas as pd
import gzip
from pathlib import Path


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


def compose():
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


def collect_classes():
    df = pd.read_json('data/processed/books_200000.json')
    categories = df.category.unique()
    categories.sort()
    d = pd.DataFrame(categories, columns=['class'])
    d = d.dropna()
    print(d)
    d.to_csv('data/processed/classes.csv', index_label='index')


def collect_labels():
    books = pd.read_json('data/processed/books_200000.json')
    classes = pd.read_csv('data/processed/classes.csv')
    d = pd.DataFrame(columns=['asin', 'class'])
    for row in books.itertuples():
        category = row.category
        entry = classes[classes['class'] == category]
        d = d.append({'asin': row.asin, 'class': entry['index']}, ignore_index=True)
        break
    return d

d = collect_labels()
print(d)