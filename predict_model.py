# get model
# predict

# create network
import os
from keras import optimizers
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

model = load_model('77_cnn.h5')

#{'batch_size': 32, 'drop_out': 0.4, 'learning_rate': 0.00625, 'freeze_layers': 150, 'momentum': 0.5654416167466141}

image_path = os.path.expanduser('~/Documents/Research/VISAGE_a/DeepUAge_dataset')

test_path = os.path.join(image_path, 'test')
datagen_batch_size = 64
batch_size = 32
image_size = 256
learning_rate = 0.00625
momentum = 0.5654416167466141


datagen = ImageDataGenerator()

# load and iterate test dataset
test_generator = datagen.flow_from_directory(
    test_path,
    class_mode='categorical',
    batch_size=batch_size,
    target_size=(image_size, image_size)
)

optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['mae'])

'''
test_image = image.load_img(imagePath, target_size = (64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#predict the result
result = model.predict(test_image)
'''

print('\n# Evaluate on test data')
results = model.evaluate_generator(test_generator, test_generator.samples)
print('test loss, test mae:', results)