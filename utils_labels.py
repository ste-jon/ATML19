import json

# assigns a book category (genre) to a number, might be useful for a simpler overview (optional?)
def idx_to_class(class_name, dict):
    for idx, name in dict.items():
        if name == class_name:
            return idx

def folder_to_cat_dict(path):
    with open('path', 'w') as outfile:
        return json.load(outfile)