from tensorflow import keras
# from tensorflow.python import keras
import numpy as np
from tensorflow.keras import backend as K


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    return keras.models.Model(input, x)
    # model = keras.models.Sequential()
    # model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
    #                               input_shape=input_shape))
    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.Conv2D(32, (3, 3)))
    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(0.25))
    #
    # model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.Conv2D(64, (3, 3)))
    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(0.25))
    #
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(512))
    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.Dropout(0.5))
    return model


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def siamese_generator(root_image, test_img_root=None, target_size=(224, 224), batch_size=32):
    train_data_gen_left = keras.preprocessing.image.ImageDataGenerator(**{"rotation_range":     40,
                                                                          "width_shift_range":  0.2,
                                                                          "height_shift_range": 0.2,
                                                                          "rescale":            1.0 / 255,
                                                                          "shear_range":        0.2,
                                                                          "zoom_range":         0.2,
                                                                          "horizontal_flip":    True,
                                                                          "fill_mode":          "nearest"})
    train_data_gen_right = keras.preprocessing.image.ImageDataGenerator(**{"rotation_range":     40,
                                                                           "width_shift_range":  0.2,
                                                                           "height_shift_range": 0.2,
                                                                           "rescale":            1.0 / 255,
                                                                           "shear_range":        0.2,
                                                                           "zoom_range":         0.2,
                                                                           "horizontal_flip":    True,
                                                                           "fill_mode":          "nearest"})
    if test_img_root is None:
        train_left_generator = train_data_gen_left.flow_from_directory(root_image, seed=7, target_size=target_size,
                                                                       batch_size=batch_size)
        train_right_generator = train_data_gen_right.flow_from_directory(root_image, seed=8, target_size=target_size,
                                                                         batch_size=batch_size)

        while True:
            left_x, left_y = next(train_left_generator)
            right_x, right_y = next(train_right_generator)
            yield [left_x, right_x], np.equal(left_y, right_y).all(axis=1).astype(int)
    else:
        train_left_generator = train_data_gen_left.flow_from_directory(root_image, seed=7, target_size=target_size)
        test_right_generator = keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255).flow_from_directory(test_img_root, seed=8, target_size=target_size)
        while True:
            try:
                right_x, right_y = next(test_right_generator)
            except StopIteration:
                return
        left_x, left_y = next(train_left_generator)
        yield [left_x, right_x], np.equal(left_y, right_y).all(axis=1).astype(int)


if __name__ == '__main__':
    dest_root_image = r'/l/workspace/dataset/asghar/train/images'
    test_root_image = r'/l/workspace/dataset/asghar/test/images'
    test_right_generator = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255).flow_from_directory(test_root_image, seed=8, target_size=(128, 128), class_mode='sparse',
                                                   batch_size=1)
    q = siamese_generator(dest_root_image, target_size=(128, 128))
    input_shape = (128, 128, 3)
    base_network = create_base_network(input_shape)

    input_a = keras.layers.Input(shape=input_shape)
    input_b = keras.layers.Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = keras.layers.Lambda(euclidean_distance,
                                   output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = keras.models.Model([input_a, input_b], distance)

    # train
    rms = keras.optimizers.Nadam()
    model.compile(loss=contrastive_loss, optimizer=rms)
    hist = model.fit_generator(q,
                               steps_per_epoch=300 * 80 / 128,
                               epochs=10
                               )
    import random
    from pathlib import Path

    t = Path(dest_root_image)
    train_sample = []
    train_label = []
    import matplotlib.pyplot as plt

    for i in t.iterdir():
        if i.is_dir():
            imgs = list(i.iterdir())
            random.shuffle(imgs)
            train_sample.append(plt.imread(imgs[0]))
            train_label.append(i.stem)
    from skimage.transform import resize

    train_sample = [resize(i, input_shape) for i in train_sample]
    train_sample = np.array(train_sample)
    train_labeled_numeric = [test_right_generator.class_indices[j] for j in train_label]
    test_l = test_right_generator.n
    counts = 0
    acc = []
    try:
        model.save_weight('siamese.h5')
    except:
        pass
    for (i, j) in test_right_generator:
        test_tmp = np.zeros_like(train_sample)
        test_tmp[:] = i[0]
        scores = model.predict([train_sample, test_tmp])
        pred_y = train_labeled_numeric[scores.argmax()]
        acc.append(pred_y == int(j[0]))
        counts += 1
        print(counts)
        if counts >= test_right_generator.n:
            break
    import pandas as pd

    h = pd.DataFrame(hist.history)
    plt.figure()
    h.plot()
    plt.save_fig('siamese.png')
    acc = sum(acc) / len(acc)
    print(acc)
