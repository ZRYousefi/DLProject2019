from pathlib import Path
from tensorflow import keras
from tensorflow.python import keras

import matplotlib.pyplot as plt
import numpy as np


def show_images(root_dir, cols=1, titles=None):
    root_dir = Path(root_dir)
    images = []
    for img in root_dir.iterdir():
        print(img)
        images.append(plt.imread(img))
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.savefig(root_dir.joinpath('tile_images.png'))


def sample_datagen(img_dir, image_data_generator, preview_img_dest_root):
    img_dir = Path(img_dir)
    preview_img_dest_root = Path(preview_img_dest_root)
    try:
        preview_img_dest_root.mkdir(parents=True)
        random_img_file = next(next(img_dir.iterdir()).iterdir())
        img = keras.preprocessing.image.load_img(random_img_file)
        img = keras.preprocessing.image.img_to_array(img)
        img = img.reshape((1,) + img.shape)
        counts = 0
        for batch in image_data_generator.flow(img, batch_size=1, save_to_dir=preview_img_dest_root, save_prefix='food_101',
                                               save_format='jpeg'):
            counts += 1
            if counts > 9:
                break
        show_images(preview_img_dest_root, cols=2)
    except FileExistsError:
        return
    except :
        return

if __name__ == '__main__':
    img_dir = r'/l/workpace/dataset/deeppg/train/images'
    dest = 'preview'
    a = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    sample_datagen(img_dir, a, dest)
