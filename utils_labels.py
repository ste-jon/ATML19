import json

# assigns a book category (genre) to a number, might be useful for a simpler overview (optional?)
def idx_to_class(index, dict):
    for name, idx in dict.items():
        if idx == index:
            return name

def folder_to_cat_dict(path):
    with open('path', 'w') as outfile:
        return json.load(outfile)