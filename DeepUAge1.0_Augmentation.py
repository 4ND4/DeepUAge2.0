import argparse
import os
from pathlib import Path

import neptune
import neptune_tensorboard as neptune_tb
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
import config
from generator import FaceGenerator, ValGenerator
from model import get_model, age_mae

image_directory = os.path.expanduser(config.image_directory)

list_images = [x for x in os.listdir(os.path.join(image_directory, 'train')) if not x.startswith('.')]
nb_classes = len(list_images)


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--pred_dir", type=str,
                        help="path to the Predator face dataset",
                        default=image_directory)

    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--batch_size", type=int, default=config.batch_size,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=config.MAXIMUM_EPOCHS,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--model_name", type=str, default="ResNet50",
                        help="model name: 'ResNet50' or 'InceptionResNetV2'")
    args = parser.parse_args()
    return args


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


def main():
    args = get_args()

    experiment_name = 'yu4u'
    early_stop_patience = config.EARLY_STOP_EPOCHS

    PARAMS = {
        'epoch_nr': args.nb_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        # 'input_shape': (512, 32, 3),
        'early_stop': early_stop_patience,
        'image_size': config.IMAGE_SIZE,
        'network': args.model_name
    }

    neptune.init(project_qualified_name='4ND4/sandbox')
    neptune_tb.integrate_with_keras()
    result = neptune.create_experiment(name=experiment_name, params=PARAMS)

    name = result.id

    print(name)

    predator_dir = args.pred_dir

    model_name = args.model_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr

    image_size = config.IMAGE_SIZE

    train_gen = FaceGenerator(predator_dir, batch_size=batch_size, image_size=image_size, number_classes=nb_classes)
    val_gen = ValGenerator(predator_dir, batch_size=batch_size, image_size=image_size, number_classes=nb_classes)

    model = get_model(model_name=model_name, image_size=image_size, number_classes=nb_classes)

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=[age_mae])

    model.summary()

    output_dir = Path(__file__).resolve().parent.joinpath(args.output_dir)

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
                               epochs=nb_epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_dir.joinpath("history_{}.npz".format(name))), history=hist.history)


if __name__ == '__main__':
    main()
