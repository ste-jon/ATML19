from PIL import Image
import os

def convert_to_jpeg(covers_dir, verbose=False):
    files = os.listdir(covers_dir)
    done = 0
    for file in files:
        filename_orig = os.path.join(covers_dir, file)
        image = Image.open(filename_orig)
        if (image.format == 'JPEG' and filename_orig.endswith('.jpeg')):
            done += 1
            continue
        if (image.format != 'JPEG'): image = image.convert('RGB')
        pre, ext = os.path.splitext(file)
        filename = os.path.join(covers_dir, pre + '.jpeg')
        image.save(filename, 'jpeg')
        os.remove(filename_orig)
        done += 1
        if verbose: print(str(done) + '/' + str(files.__len__()))
    print('Done')