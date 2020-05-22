import os
import keras
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
from keras import Model, optimizers, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense, AveragePooling2D
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

import config

maximum_epochs = config.MAXIMUM_EPOCHS
early_stop_epochs = config.EARLY_STOP_EPOCHS
learning_rate_epochs = config.LEARNING_RATE_EPOCHS
optimizer_direction = config.OPTIMIZER_DIRECTION
results_directory = config.RESULTS_DIRECTORY

image_path = os.path.expanduser(config.IMAGE_PATH)
log_results = config.LOG_RESULTS
image_size = config.IMAGE_SIZE


class Objective(object):
    def __init__(
            self,
            # train_generator, valid_generator, test_generator,
            dir_save,
            max_epochs, early_stop, learn_rate_epochs,
            # input_shape,
            # number_of_classes
    ):
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.dir_save = dir_save
        self.learn_rate_epochs = learn_rate_epochs

    def __call__(self, trial):

        # SET PARAMETERS HERE

        batch_size = 32
        unit = 256
        drop_out = 0.35000000000000003
        learning_rate = 0.0025389437553681262
        momentum = 0.2753813367505939

        # SET PARAMETERS HERE

        train_path = os.path.join(image_path, 'train')
        validation_path = os.path.join(image_path, 'validation')
        test_path = os.path.join(image_path, 'test')

        train_gen, val_gen, test_gen = getdata(trial, train_path, validation_path, test_path)

        # implement resnet50

        base_model = keras.applications.resnet.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=Input(shape=(224, 224, 3))

        )

        # start - changed

        # add fresh layer

        head_model = base_model.output
        head_model = AveragePooling2D(pool_size=(7, 7))(head_model)  # implement optimization later
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(unit, activation="relu")(head_model)  # implement optimization later
        head_model = Dropout(drop_out)(head_model)
        head_model = Dense(train_gen.num_classes, activation="softmax")(head_model)

        model = Model(inputs=base_model.input, outputs=head_model)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True),
                      metrics=['mae'])
        model.summary()

        # callbacks for early stopping and for learning rate reducer
        fn = self.dir_save + str(trial.number) + '_rn50_A.h5'
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=self.early_stop, verbose=1),
                          ModelCheckpoint(filepath=fn,
                                          monitor='val_loss', save_best_only=True)
                          ]

        h = model.fit_generator(
            train_gen,
            validation_data=val_gen,
            steps_per_epoch=train_gen.samples // batch_size, epochs=maximum_epochs,
            callbacks=callbacks_list,
        )

        validation_loss = np.min(h.history['val_loss'])

        return validation_loss


def getdata(trial, train_path, val_path, test_path):
    # create a data generator

    datagen_batch_size = hyperconfig.BATCH_SIZE

    horizontal_flip = trial.suggest_categorical('horizontal_flip', [True, False])
    brightness_range = trial.suggest_categorical('brightness_range', [None, [0.2, 1.0]])
    rotation_range = trial.suggest_categorical('rotation_range', [None, 30])
    zoom_range = trial.suggest_categorical('zoom_range', [0, [0.5, 1.0]])
    shift = trial.suggest_categorical('shift', [None, 0.2])
    shear_range = trial.suggest_categorical('shear_range', [None, 0.2])

    datagen = ImageDataGenerator(
        horizontal_flip=horizontal_flip,
        brightness_range=brightness_range,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=shift,  # distortion
        height_shift_range=shift,  # distortion
        shear_range=shear_range,  # distortion

    )

    # test data shouldn't be augmented

    test_datagen = ImageDataGenerator()

    train_it = datagen.flow_from_directory(
        train_path,
        class_mode='categorical',
        batch_size=datagen_batch_size,
        target_size=(image_size, image_size)
    )
    # load and iterate validation dataset
    val_it = datagen.flow_from_directory(
        val_path, class_mode='categorical', batch_size=datagen_batch_size, target_size=(image_size, image_size)
    )
    # load and iterate test dataset
    test_it = test_datagen.flow_from_directory(
        test_path, class_mode='categorical', batch_size=datagen_batch_size, target_size=(image_size, image_size))

    return train_it, val_it, test_it


objective = Objective(
    results_directory,
    maximum_epochs, early_stop_epochs,
    learning_rate_epochs
)

if log_results:

    neptune.init(project_qualified_name='4ND4/sandbox')
    result = neptune.create_experiment(name='optuna Resnet50 DeepUAge2.0')
    monitor = opt_utils.NeptuneMonitor()
    callback = [monitor]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
else:
    callback = None

study = optuna.create_study(direction=optimizer_direction,
                            # sampler=TPESampler(n_startup_trials=number_of_random_points) read paper
                            )

study.optimize(
    objective,
    callbacks=callback,
    n_trials=100
)

# save results
df_results = study.trials_dataframe()
df_results.to_pickle(results_directory + 'df_optuna_results.pkl')
df_results.to_csv(results_directory + 'df_optuna_results.csv')

print('Minimum error: ' + str(study.best_value))
print('Best parameter: ' + str(study.best_params))
print('Best trial: ' + str(study.best_trial))
