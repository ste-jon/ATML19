import json
import gzip


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))


f = open("data/meta_Books_strict.json", 'w')
for l in parse("data/meta_Books.json.gz"):
    f.write(l + '\n')
