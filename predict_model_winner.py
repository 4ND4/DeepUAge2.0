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

model_name = '1_rn50'

model = load_model('{}.h5'.format(model_name))

# {'batch_size': 32, 'drop_out': 0.4, 'learning_rate': 0.00625, 'freeze_layers': 150, 'momentum': 0.5654416167466141}

image_path = os.path.expanduser('~/Documents/Research/VISAGE_a/DeepUAge_dataset')
test_path = os.path.join(image_path, 'test')
datagen_batch_size = 32
batch_size = 32
image_size = 224
learning_rate = 0.0025389437553681262
momentum = 0.2753813367505939

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
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['mae'])

true_labels = test_generator.classes

scores = model.evaluate_generator(test_generator, test_generator.samples)
print("Error = ", scores)

predictions = model.predict_generator(test_generator)

y_true = true_labels
y_pred = np.array([np.argmax(x) for x in predictions])

# get values and copy to

with open('evaluation_{}.csv'.format(model_name), 'w') as f1:
    writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )

    for i in range(0, len(y_pred)):
        # print(y_pred[i], y_true[i])
        row = [y_pred[i], y_true[i]]
        writer.writerow(row)

cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(cm, range(20), range(20))

sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

plt.savefig('confusion_matrix_{}.png'.format(model_name))

plt.show()
