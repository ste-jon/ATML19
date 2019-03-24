import json
import gzip


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))


f = open("data/output.strict", 'w')
for l in parse("data/reviews_Books_5.json.gz"):
    f.write(l + '\n')