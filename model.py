# import better_exceptions
from keras.applications import ResNet50, InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
from keras import backend as K


def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 20, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 20, dtype="float32"), axis=-1)

    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def get_model(model_name="ResNet50", image_size=224, number_classes=20):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3),
                              pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3),
                                       pooling="avg")

    prediction = Dense(units=number_classes, kernel_initializer="he_normal", use_bias=False, activation='softmax',
                       name="pred_age")(base_model.layers[-1].output)

    model = Model(inputs=base_model.input, outputs=prediction)

    return model


def main():
    model = get_model("InceptionResNetV2")
    model.summary()


if __name__ == '__main__':
    main()
