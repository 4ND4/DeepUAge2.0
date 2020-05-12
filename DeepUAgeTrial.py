import os

import keras
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
from keras import Model, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, History
from keras.layers import Dropout, Flatten, Dense, Softmax
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from optuna.samplers import TPESampler

maximum_epochs = 1000
early_stop_epochs = 10
learning_rate_epochs = 5
optimizer_direction = 'minimize'
number_of_random_points = 25  # random searches to start opt process
maximum_time = 4 * 60 * 60  # seconds
results_directory = 'output/'

num_classes = 1
VECTOR_SIZE = 512
FACE_DETECTION = False
channel = 3
image_path = os.path.expanduser('~/Documents/Research/VISAGE_a/DeepUAge_dataset')
log_results = True


class Objective(object):
    def __init__(self, train_gen, valid_gen, test_gen, dir_save,
                 max_epochs, early_stop, learn_rate_epochs,
                 input_shape, number_of_classes):
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.test_gen = test_gen
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.dir_save = dir_save
        self.learn_rate_epochs = learn_rate_epochs
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

    def __call__(self, trial):
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
        drop_out = trial.suggest_discrete_uniform('drop_out', 0.05, 0.5, 0.05)
        learning_rate = trial.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.00025)
        freeze_layers = trial.suggest_categorical('freeze_layers', [15, 25, 32, 40, 100, 150])
        momentum = trial.suggest_uniform('momentum', 0, 1.0)

        # implement resnet50

        resnet_50 = keras.applications.resnet.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
        )

        # start - changed

        x = Flatten()(resnet_50.output)

        x = Dropout(drop_out)(x)

        predictions = Dense(
            units=20,
            kernel_initializer="he_normal",
            use_bias=False,
            activation="softmax"
        )(x)

        model = Model(inputs=resnet_50.inputs, outputs=predictions)

        for layer in model.layers[:-freeze_layers]:
            layer.trainable = False

        model.compile(loss='mse',
                      optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True),
                      metrics=['mae'])
        model.summary()

        # callbacks for early stopping and for learning rate reducer
        fn = self.dir_save + str(trial.number) + '_cnn.h5'
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=self.early_stop),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                            patience=self.learn_rate_epochs,
                                            verbose=1, mode='auto', min_lr=1.0e-6),
                          ModelCheckpoint(filepath=fn,
                                          monitor='val_loss', save_best_only=True)
                          ]

        # fit the model
        # h = model.fit(x=self.xcalib, y=self.ycalib,
        #              batch_size=batch_size,
        #              epochs=self.max_epochs,
        #              validation_data=(self.xvalid, self.yvalid),
        #              shuffle=True, verbose=1,
        #              callbacks=callbacks_list)

        h = model.fit_generator(
            train_gen, train_gen.samples // batch_size, callbacks=callbacks_list,
            validation_data=val_gen, validation_steps=val_gen.samples // batch_size,
            epochs=maximum_epochs
        )

        validation_loss = np.min(h.history['val_loss'])

        return validation_loss


def getdata(train_path, val_path, test_path):
    # create a data generator
    datagen = ImageDataGenerator()
    train_it = datagen.flow_from_directory(train_path, class_mode='categorical', batch_size=64)
    # load and iterate validation dataset
    val_it = datagen.flow_from_directory(val_path, class_mode='categorical', batch_size=64)
    # load and iterate test dataset
    test_it = datagen.flow_from_directory(test_path, class_mode='categorical', batch_size=64)

    return train_it, val_it, test_it


train_path = os.path.join(image_path, 'train')
validation_path = os.path.join(image_path, 'validation')
test_path = os.path.join(image_path, 'test')

train_gen, val_gen, test_gen = getdata(train_path, validation_path, test_path)

shape_of_input = train_gen.image_shape


objective = Objective(train_gen, val_gen, test_gen, results_directory,
                      maximum_epochs, early_stop_epochs,
                      learning_rate_epochs, shape_of_input, num_classes)


if log_results:

    neptune.init(project_qualified_name='4ND4/sandbox')
    result = neptune.create_experiment(name='optuna Resnet50 DeepUAge')
    monitor = opt_utils.NeptuneMonitor()
    callback = [monitor]
    n_trials = 100
    optuna.logging.set_verbosity(optuna.logging.WARNING)
else:
    callback = None

study = optuna.create_study(direction=optimizer_direction,
                            sampler=TPESampler(n_startup_trials=number_of_random_points))

study.optimize(
    objective,
    timeout=maximum_time,
    callbacks=callback
)

# save results
df_results = study.trials_dataframe()
df_results.to_pickle(results_directory + 'df_optuna_results.pkl')
df_results.to_csv(results_directory + 'df_optuna_results.csv')

print('Minimum error: ' + str(study.best_value))
print('Best parameter: ' + str(study.best_params))
print('Best trial: ' + str(study.best_trial))
