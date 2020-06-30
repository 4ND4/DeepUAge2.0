import os
from pathlib import Path
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import config
from generator import FaceGenerator, ValGenerator
from model import get_model, age_mae

image_directory = os.path.expanduser(config.image_directory)
list_images = [x for x in os.listdir(os.path.join(image_directory, 'train')) if not x.startswith('.')]
nb_classes = len(list_images)
max_epochs = config.MAXIMUM_EPOCHS
log_results = config.LOG_RESULTS
early_stop_epochs = config.EARLY_STOP_EPOCHS
learning_rate_epochs = config.LEARNING_RATE_EPOCHS
optimizer_direction = 'minimize'
name = 'debug'


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


if log_results:
    neptune.init(project_qualified_name='4ND4/sandbox')
    result = neptune.create_experiment(name='optuna DeepUAge1.0')
    monitor = opt_utils.NeptuneMonitor()
    callback = [monitor]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    name = result.id
else:
    callback = None
    name = 'debug'


class Objective(object):
    def __init__(
            self, dir_save,
            nb_epochs, early_stop, learn_rate_epochs,
            number_of_classes):
        self.max_epochs = nb_epochs
        self.early_stop = early_stop
        self.dir_save = dir_save
        self.learn_rate_epochs = learn_rate_epochs
        self.number_of_classes = number_of_classes

    def __call__(self, trial):

        batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        momentum = trial.suggest_uniform('momentum', 0, 1.0)
        image_size = trial.suggest_categorical('image_size', [64, 128, 224])
        model_name = trial.suggest_categorical('model_name', ['ResNet50', 'InceptionResNetV2'])

        model = get_model(model_name=model_name, image_size=image_size, number_classes=nb_classes)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True),
                      metrics=[age_mae])
        model.summary()

        # callbacks for early stopping and for learning rate reducer

        output_dir = Path(__file__).resolve().parent.joinpath(self.dir_save)

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        if not os.path.exists('{}/{}'.format(output_dir, name)):
            os.mkdir('{}/{}'.format(output_dir, name))

        if not os.path.exists('{}/{}/{}'.format(output_dir, name, str(trial.number))):
            os.mkdir('{}/{}/{}'.format(output_dir, name, str(trial.number)))

        fn = os.path.join(self.dir_save, "{}/{}".format(name, str(trial.number)), "weights.{epoch:03d}-{val_loss:.3f}-{"
                                                                                  "val_age_mae:.3f}.hdf5")

        callbacks_list = [EarlyStopping(monitor='val_loss', patience=self.early_stop, verbose=1),
                          ModelCheckpoint(filepath=fn,
                                          monitor='val_loss', save_best_only=True)
                          ]

        train_gen, valid_gen = get_data(image_directory, batch_size, image_size)

        h = model.fit_generator(
            train_gen,
            validation_data=valid_gen,
            epochs=max_epochs,
            callbacks=callbacks_list,
        )

        validation_loss = np.min(h.history['val_loss'])

        return validation_loss


def get_data(path, batch_size, image_size):
    train_generator = FaceGenerator(
        path, batch_size=batch_size, image_size=image_size, number_classes=nb_classes
    )

    val_generator = ValGenerator(
        path, batch_size=batch_size, image_size=image_size, number_classes=nb_classes
    )

    return train_generator, val_generator


results_directory = 'output/'

objective = Objective(results_directory,
                      max_epochs, early_stop_epochs,
                      learning_rate_epochs, nb_classes)

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
