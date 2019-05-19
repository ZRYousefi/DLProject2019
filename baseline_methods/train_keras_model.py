import tensorflow as tf
from pathlib import Path
from constants import RANDOM_SEED
from tensorflow.python import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.utils import plot_model
import logging
import sys
from utils import counts_file
import pandas as pd
import numpy as np
from baseline_methods.plot_utils import sample_datagen

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
fileh = logging.FileHandler("logfile", "a")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fileh.setFormatter(formatter)
log = logging.getLogger("root")
log.addHandler(fileh)
logger = logging.getLogger("root")
import json


def init_experiment(expr_root_dir):
    tf.random.set_seed(RANDOM_SEED)
    expr_root_dir = Path(expr_root_dir)
    expr_root_dir.mkdir(exist_ok=True, parents=True)
    dir_counts = str(len(list(expr_root_dir.iterdir())))
    expr_root_dir.joinpath(dir_counts).mkdir(parents=True)
    return expr_root_dir.joinpath(dir_counts)


def get_model(m):
    models = {"Xception": Xception, "VGG16": VGG16, "Resnet": ResNet50,"MobileNet":MobileNet}
    return models[m]


def train_keras_baseline(
        expr_root_dir,
        train_img_dir,
        val_img_dir,
        test_img_dir=None,
        n_epochs=100,
        which_model="xception",
        image_augmentaiton_params={
            "rotation_range": 40,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "rescale": 1.0 / 255,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "horizontal_flip": True,
            "fill_mode": "nearest",
        },
        batch_size=9,
        optimzier="nadam",
        init_model_weight=None,
):
    logger.info("*" * 10, f"New expr: {expr_root_dir}", "*" * 10)
    this_expr_root = init_experiment(Path(expr_root_dir).joinpath(which_model))
    train_img_dir = Path(train_img_dir)
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        **image_augmentaiton_params)
    sample_datagen(
        train_img_dir,
        train_datagen,
        preview_img_dest_root=Path(expr_root_dir).joinpath("preview_images"),
    )
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 /
                                                                255)
    train_data_count = counts_file(train_img_dir)
    val_data_count = counts_file(val_img_dir)
    test_data_count = counts_file(test_img_dir)
    nb_classes = len(list(train_img_dir.iterdir()))
    base_model = get_model(which_model)(weights="imagenet", include_top=False)
    x = base_model.output
   

    # add your top layer block to your base model
    if which_model=='Xception' or which_model=='MobileNet':

        for layer in base_model.layers[:-4]:
                layer.trainable = False
        x = keras.layers.GlobalAveragePooling2D()(x)
        predictions = keras.layers.Dense(nb_classes, activation="softmax")(x)
        # Plot model
    elif which_model=='VGG16':
            # build a classifier model to put on top of the convolutional model
        
        x = keras.layers.Flatten(input_shape=base_model.output_shape[1:])(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        predictions = keras.layers.Dense(nb_classes, activation='sigmoid')(x)

        for layer in base_model.layers[:25]:
                layer.trainable = False
    model = keras.models.Model(base_model.input, predictions)
    plot_model(model,
               to_file=this_expr_root.joinpath("model_architecture.png"))

    train_generator = train_datagen.flow_from_directory(
        train_img_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
    )
    val_generator = validation_datagen.flow_from_directory(
        val_img_dir, batch_size=batch_size)
    test_generator = test_datagen.flow_from_directory(test_img_dir,
                                                      batch_size=batch_size)
    top_weights_path = this_expr_root.joinpath(
        "top_model_weights.h5").__str__()
    if init_model_weight is not None and Path(init_model_weight).exists():
        model.load_weights(str(init_model_weight))
        print(f"{init_model_weight} is loaded")
    model.compile(optimizer=optimzier,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(top_weights_path,
                                        monitor="val_accuracy",
                                        verbose=1,
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                      patience=5,
                                      verbose=0),
        keras.callbacks.TensorBoard(
            log_dir=str(this_expr_root.joinpath("tensorboard"))),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.2,
                                          patience=5,
                                          min_lr=0.001)
    ]

    # Train Simple CNN
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_data_count // batch_size,
        epochs=n_epochs,
        callbacks=callbacks_list,
        validation_data=val_generator,
        validation_steps=val_data_count // batch_size,
        use_multiprocessing=True,
        workers=6)
    model.save_weights(
        filepath=str(this_expr_root.joinpath("final_weights.h5")))

    # save model
    model_json = model.to_json()
    history = pd.DataFrame(history.history)
    with open(this_expr_root.joinpath("model.json"), "w") as json_file:
        json_file.write(model_json)
    if init_model_weight is not None and Path(init_model_weight).exists():
        init_model_weight = Path(init_model_weight)
        previous_history_path = init_model_weight.parent.joinpath(
            "history.json")
        previous_history = pd.read_json(previous_history_path)
        history = pd.concat((previous_history, history),
                            axis=0,
                            ignore_index=True)

    test_loss, test_acc = model.evaluate_generator(test_generator,
                                                   steps=test_data_count //
                                                   batch_size)
    json.dump(
        {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc)
        },
        open(this_expr_root.joinpath("test_stats.json"), "w"),
    )

    history.to_json(this_expr_root.joinpath("history.json"))

    plot_stats(history, this_expr_root)
    keras.backend.clear_session()
    del model

#     try:
#         from numba import cuda
#         cuda.select_device(0)
#         cuda.close()
#     except:
#         pass

    return str(this_expr_root.joinpath("final_weights.h5"))


def plot_stats(df, dest_root):
    import matplotlib.pyplot as plt
    plt.figure()
    lr = df.loc[:, 'lr']
    fig = lr.plot()
    plt.savefig(dest_root.joinpath("lr.png"))
    plt.figure()
    loss_data = df.loc[:, df.columns.str.contains('loss')]
    fig = loss_data.plot()
    plt.savefig(dest_root.joinpath("loss.png"))
    plt.figure()
    acc_data = df.loc[:, df.columns.str.contains('accuracy')]
    fig = acc_data.plot()
    plt.savefig(dest_root.joinpath("acc.png"))
    plt.close('all')