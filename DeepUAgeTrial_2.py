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

maximum_epochs = 1000
early_stop_epochs = 15
learning_rate_epochs = 5
optimizer_direction = 'minimize'
results_directory = 'output/'

num_classes = 20
VECTOR_SIZE = 512
channel = 3
image_path = os.path.expanduser('~/Documents/Research/VISAGE_a/DeepUAge_dataset')
log_results = False
image_size = 224


class Objective(object):
    def __init__(
            self, train_generator, valid_generator, test_generator, dir_save,
            max_epochs, early_stop, learn_rate_epochs,
            input_shape, number_of_classes):
        self.train_gen = train_generator
        self.valid_gen = valid_generator
        self.test_gen = test_generator
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.dir_save = dir_save
        self.learn_rate_epochs = learn_rate_epochs
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

    def __call__(self, trial):
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
        unit = trial.suggest_categorical('unit', [2048, 1024, 512, 256])
        drop_out = trial.suggest_discrete_uniform('drop_out', 0.05, 0.5, 0.05)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        momentum = trial.suggest_uniform('momentum', 0, 1.0)

        # implement resnet50

        base_model = keras.applications.resnet.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=Input(shape=self.input_shape)
        )

        # start - changed

        # add fresh layer

        head_model = base_model.output
        head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(unit, activation="relu")(head_model)  # implement optimization later
        head_model = Dropout(drop_out)(head_model)
        head_model = Dense(self.number_of_classes, activation="softmax")(head_model)

        model = Model(inputs=base_model.input, outputs=head_model)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True),
                      metrics=['mae'])
        model.summary()

        # callbacks for early stopping and for learning rate reducer
        fn = self.dir_save + str(trial.number) + '_rn50.h5'
        callbacks_list = [EarlyStopping(monitor='val_mae', patience=self.early_stop, verbose=1),
                          ModelCheckpoint(filepath=fn,
                                          monitor='val_mae', save_best_only=True)
                          ]

        h = model.fit_generator(
            train_gen,
            validation_data=val_gen,
            steps_per_epoch=train_gen.samples // batch_size, epochs=maximum_epochs,
            callbacks=callbacks_list,
        )

        validation_loss = np.min(h.history['val_loss'])

        return validation_loss


def getdata(train_path, val_path, test_path):
    # create a data generator

    datagen_batch_size = 64

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


train_ds_path = os.path.join(image_path, 'train')
validation_ds_path = os.path.join(image_path, 'validation')
test_ds_path = os.path.join(image_path, 'test')

train_gen, val_gen, test_gen = getdata(train_ds_path, validation_ds_path, test_ds_path)

shape_of_input = train_gen.image_shape

objective = Objective(train_gen, val_gen, test_gen, results_directory,
                      maximum_epochs, early_stop_epochs,
                      learning_rate_epochs, shape_of_input, num_classes)

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
