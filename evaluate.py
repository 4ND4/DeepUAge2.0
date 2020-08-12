# get model
# predict

# create network
import os

import neptune
import sklearn.metrics
import numpy as np
from keras import optimizers
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import csv
from model import age_mae
import config

# log the evaluation


log_results = config.LOG_RESULTS

weights_filename = 'weights.087-1.799-1.528'

experiment_name = 'SAN-278'

model = load_model('checkpoints/{}/{}.hdf5'.format(experiment_name, weights_filename), custom_objects={'age_mae': age_mae})

image_path = os.path.expanduser(config.IMAGE_PATH)
test_path = os.path.join(image_path, 'test')
batch_size = 64
image_size = 224
learning_rate = 0.1
momentum = 0.9


PARAMS = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'momentum': momentum,
        'image_size': image_size,
        'test_path': image_path
    }

if log_results:

    neptune.init(project_qualified_name='4ND4/sandbox')
    #neptune_tb.integrate_with_keras()
    result = neptune.create_experiment(name='evaluation_{}'.format(experiment_name), params=PARAMS, tags='evaluation')
    name = result.id
else:
    name = 'debug'

datagen = ImageDataGenerator()

# load and iterate test dataset
test_generator = datagen.flow_from_directory(
    test_path,
    class_mode='categorical',
    batch_size=batch_size,
    target_size=(image_size, image_size),
    shuffle=False

)

optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[age_mae])

true_labels = test_generator.classes

scores = model.evaluate_generator(test_generator, test_generator.samples)
MAE = scores[1]

print("MAE = ", MAE)

neptune.log_metric('mean_absolute_error', MAE)

predictions = model.predict_generator(test_generator)

y_true = true_labels
y_pred = np.array([np.argmax(x) for x in predictions])

# get values and copy to

with open('evaluation_{}_{}.csv'.format(name, weights_filename), 'w') as f1:
    writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )

    for i in range(0, len(y_pred)):
        row = [y_pred[i], y_true[i]]
        writer.writerow(row)

cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(cm, range(20), range(20))

sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

plt.savefig('confusion_matrix_{}.png'.format(weights_filename))

plt.show()
