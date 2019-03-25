import array


def read_image_features(path):
    f = open(path, 'rb')
    while True:
        asin = f.read(10)
        if asin == '': break
        a = array.array('f')
        a.fromfile(f, 4096)
        yield asin, a.tolist()
