from PIL import Image
import os

dir_path = "../data/coversraw/"
files = os.listdir(dir_path)
done = 0
for file in files:
    filename_orig = os.path.join(dir_path, file)
    image = Image.open(filename_orig)
    if (image.format == 'JPEG' and filename_orig.endswith('.jpeg')):
        done +=1
        continue
    if (image.format != 'JPEG'): image = image.convert('RGB')
    pre, ext = os.path.splitext(file)
    filename = os.path.join(dir_path, pre + '.jpeg')
    image.save(filename, 'jpeg')
    os.remove(filename_orig)
    done += 1
    print(str(done) + '/' + str(files.__len__()))