import os
from pathlib import Path

import Augmentor
import neptune
import neptune_tensorboard as neptune_tb
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras_applications import xception
from keras_preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

import config
from model import get_model, age_mae

image_directory = os.path.expanduser(config.image_directory)
log_experiment = False

list_images = [x for x in os.listdir(os.path.join(image_directory, 'train')) if not x.startswith('.')]
nb_classes = len(list_images)


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def getdata(train_path, val_path, test_path):
    # create a data generator

    image_size = config.IMAGE_SIZE

    datagen_batch_size = config.batch_size

    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)

    #train_it = p.keras_generator(batch_size=datagen_batch_size, image_data_format='channels_last')

    train_datagen = ImageDataGenerator(
        #rescale=1. / 255,
        preprocessing_function=p.keras_preprocess_func()
    )

    # test data shouldn't be augmented

    val_datagen = ImageDataGenerator(
        #rescale=1./255
    )

    test_datagen = ImageDataGenerator(
        #rescale=1./255
    )

    train_it = train_datagen.flow_from_directory(
        train_path, class_mode='categorical', batch_size=datagen_batch_size, target_size=(image_size, image_size)
    )

    # load and iterate validation dataset
    val_it = val_datagen.flow_from_directory(
        val_path, class_mode='categorical', batch_size=datagen_batch_size, target_size=(image_size, image_size)
    )
    # load and iterate test dataset
    test_it = test_datagen.flow_from_directory(
        test_path, class_mode='categorical', batch_size=datagen_batch_size, target_size=(image_size, image_size))

    return train_it, val_it, test_it, train_it.samples


def main():

    nb_epochs = config.MAXIMUM_EPOCHS
    batch_size = config.batch_size
    lr = 0.1
    momentum = 0.9
    model_name = 'ResNet50'
    image_size = config.IMAGE_SIZE
    output_dir = 'checkpoints'

    experiment_name = 'data_augmentation'
    early_stop_patience = config.EARLY_STOP_EPOCHS

    train_path = os.path.join(image_directory, 'train')
    validation_path = os.path.join(image_directory, 'validation')
    test_path = os.path.join(image_directory, 'test')

    PARAMS = {
        'epoch_nr': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'momentum': momentum,
        # 'input_shape': (512, 32, 3),
        'early_stop': early_stop_patience,
        'image_size': image_size,
        'network': model_name
    }

    if log_experiment:
        neptune.init(project_qualified_name='4ND4/sandbox')
        neptune_tb.integrate_with_keras()
        result = neptune.create_experiment(name=experiment_name, params=PARAMS)

        name = result.id

        print(name)
    else:
        name = 'debug'

    train_gen, val_gen, test_gen, len_train = getdata(train_path, validation_path, test_path)

    model = get_model(model_name=model_name, image_size=image_size, number_classes=nb_classes)

    sgd = SGD(lr=lr, momentum=momentum, nesterov=True)

    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=[age_mae])

    model.summary()

    output_dir = Path(__file__).resolve().parent.joinpath(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if not os.path.exists('checkpoints/{}'.format(name)):
        os.mkdir('checkpoints/{}'.format(name))

    callbacks = [EarlyStopping(monitor='val_age_mae', mode='min', verbose=1, patience=early_stop_patience),
                 LearningRateScheduler(schedule=Schedule(nb_epochs, initial_lr=lr)),
                 ModelCheckpoint(os.path.join(output_dir, name) + "/weights.{epoch:03d}-{val_loss:.3f}-{"
                                                                  "val_age_mae:.3f}.hdf5",
                                 monitor="val_age_mae",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min")
                 ]

    hist = model.fit_generator(generator=train_gen,
                               steps_per_epoch=len_train // batch_size,
                               validation_data=val_gen,
                               epochs=nb_epochs, verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_dir.joinpath("history_{}.npz".format(name))), history=hist.history)


if __name__ == '__main__':
    main()
