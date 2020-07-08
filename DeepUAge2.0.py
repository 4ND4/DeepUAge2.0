import os
from pathlib import Path
import neptune
import neptune_tensorboard as neptune_tb
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator

import config
from model import get_model, age_mae

image_directory = os.path.expanduser(config.image_directory)

list_images = [x for x in os.listdir(os.path.join(image_directory, 'train')) if not x.startswith('.')]
nb_classes = len(list_images)

# monitor = 'val_age_mae'
monitor = 'val_loss'


def getdata(train_path, val_path, test_path):
    # create a data generator

    datagen_batch_size = config.batch_size
    image_size = config.IMAGE_SIZE

    datagen = ImageDataGenerator()
    train_it = datagen.flow_from_directory(
        train_path, class_mode='categorical', batch_size=datagen_batch_size, target_size=(image_size, image_size)
    )
    # load and iterate validation dataset
    val_it = datagen.flow_from_directory(
        val_path, class_mode='categorical', batch_size=datagen_batch_size, target_size=(image_size, image_size)
    )
    # load and iterate test dataset
    test_it = datagen.flow_from_directory(
        test_path, class_mode='categorical', batch_size=datagen_batch_size, target_size=(image_size, image_size))

    return train_it, val_it, test_it


def main():
    nb_epochs = config.MAXIMUM_EPOCHS
    batch_size = config.batch_size
    lr = 0.1
    momentum = 0.9
    model_name = 'ResNet50'
    image_size = config.IMAGE_SIZE
    output_dir = 'checkpoints'

    experiment_name = 'DeepUAge2.0'
    early_stop_patience = config.EARLY_STOP_EPOCHS

    train_path = os.path.join(image_directory, 'train')
    validation_path = os.path.join(image_directory, 'validation')
    test_path = os.path.join(image_directory, 'test')

    PARAMS = {
        'epoch_nr': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'momentum': momentum,
        'early_stop': early_stop_patience,
        'image_size': image_size,
        'network': model_name
    }

    neptune.init(project_qualified_name='4ND4/sandbox')
    neptune_tb.integrate_with_keras()
    result = neptune.create_experiment(name=experiment_name, params=PARAMS)

    name = result.id

    print(name)

    train_gen, val_gen, test_gen = getdata(train_path, validation_path, test_path)

    model = get_model(model_name=model_name, image_size=image_size, number_classes=nb_classes)

    sgd = SGD(lr=lr, momentum=momentum, nesterov=True)

    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=[age_mae])

    model.summary()

    output_dir = Path(__file__).resolve().parent.joinpath(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if not os.path.exists('checkpoints/{}'.format(name)):
        os.mkdir('checkpoints/{}'.format(name))

    callbacks = [
        EarlyStopping(monitor=monitor, mode='min', verbose=1, patience=early_stop_patience),
        ModelCheckpoint(os.path.join(output_dir, name) + "/weights.{epoch:03d}-{val_loss:.3f}-{"
                                                         "val_age_mae:.3f}.hdf5",
                        monitor=monitor,
                        verbose=1,
                        save_best_only=True,
                        mode="min")
    ]

    hist = model.fit_generator(generator=train_gen,
                               epochs=nb_epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_dir.joinpath("history_{}.npz".format(name))), history=hist.history)


if __name__ == '__main__':
    main()
