# get model
# predict

# create network
import os
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


weights_filename = 'weights.030-2.360-1.540'

model = load_model('models/{}.hdf5'.format(weights_filename), custom_objects={'age_mae': age_mae})

image_path = os.path.expanduser(config.IMAGE_PATH)
test_path = os.path.join(image_path, 'test')
datagen_batch_size = 64
batch_size = 64
image_size = 224
learning_rate = 0.1
momentum = 0.9

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
print("MAE = ", scores[1])

predictions = model.predict_generator(test_generator)

y_true = true_labels
y_pred = np.array([np.argmax(x) for x in predictions])

# get values and copy to

with open('evaluation_{}.csv'.format(weights_filename), 'w') as f1:
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
