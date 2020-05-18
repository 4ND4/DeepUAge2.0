# imports
import keras
import neptune
import neptune_tensorboard as neptune_tb
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, Normalizer

DEBUG = False
LOG_NEPTUNE = True
EPOCHS = 1000

if DEBUG:
    EPOCHS = 100

# parameters

FACE_DETECTION = False

PARAMS = {
    'epoch_nr': EPOCHS,
    'batch_size': 64,
    'learning_rate': 0.006,
    'input_shape': (512, 32, 3),
    'early_stop': 20
}

# start experiment

name = 'resnet50-experiment'

if LOG_NEPTUNE:
    neptune.init(project_qualified_name='4ND4/sandbox')
    neptune_tb.integrate_with_keras()
    result = neptune.create_experiment(name=name, params=PARAMS)

    name = result.id


def getdata():



    return trainX, trainy, valX, valy, test_x, test_y


# start of ResNet50 coding

resnet_50 = keras.applications.resnet.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=PARAMS.get('input_shape'),
)

x = Flatten()(resnet_50.output)

predictions = Dense(1)(x)

model = Model(inputs=resnet_50.inputs, outputs=predictions)

model.compile(loss='mse',
              optimizer=Adam(lr=PARAMS.get('learning_rate')),
              metrics=['mae'])

model.summary()

# callbacks for early stopping and for learning rate reducer

callbacks_list = [EarlyStopping(monitor='val_loss', patience=PARAMS.get('early_stop')),
                  ModelCheckpoint(
                      filepath='checkpoint/model-{epoch:03d}-{val_loss:03f}-{val_mae:03f}.h5',
                      monitor='val_loss', save_best_only=True, verbose=1)
                  ]

train_X, train_Y, val_X, val_Y, test_X, test_Y = getdata()

# fit the model

#h = model.fit(x=train_X, y=train_Y,
#              batch_size=PARAMS.get('batch_size'),
#              epochs=PARAMS.get('epoch_nr'),
##              validation_data=(val_X, val_Y),
#              shuffle=True, verbose=1,
#              callbacks=callbacks_list)



h = model.fit_generator(
    train_gen, train_gen.samples // batch_size, callbacks=callbacks_list,
    validation_data=val_gen, validation_steps=val_gen.samples // batch_size,
    epochs=EPOCHS
)

validation_loss = np.min(h.history['val_loss'])

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
resun;jew.lts = model.evaluate(test_X, test_Y, batch_size=128)
print('test loss, test mae:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = model.predict(test_X[:3])
print('predictions shape:', predictions.shape)
model.save('model/model_{}.h5'.format(name))
