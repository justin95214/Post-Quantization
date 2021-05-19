#install tensorflow 2.2 or 1.15.0
#install tensorflow-gpu 2.2 or 1.15.0
#base on VGG16
#image size >> (96,72)

from __future__ import print_function
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
from sklearn.model_selection import train_test_split
import json
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

def cnn():

    rows, cols = 96, 72

    #    x_data, y_data = load_data(rows, cols)

    # x_data = np.array(x_data)  # 462
    # y_data = np.array(y_data)
    x_data = np.load('D:/download/X0.npy')  # 462
    y_data = np.load('D:/download/y0_data.npy')

    # x_data = np.reshape(x_data,(x_data.shape + (1,)))
    # y_data = np.reshape(y_data,(y_data.shape + (7,)))
    # y_data = np.expand_dims(y_data,axis=)

    print(x_data.shape)
    print(y_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data, random_state=0)

    batch_size = 32
    num_classes = 41
    epochs = 1
    earlystopping = EarlyStopping(monitor="val_accuracy", patience=20)

    # input image dimensions
    img_rows, img_cols = rows, cols

    input_shape = (img_rows, img_cols, 3)

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # model.add(Dense(512, activation="relu"))
    # model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.003),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[earlystopping])

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('test6.h5')
    print("save the model")


    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    #################################################################################3
    import tempfile
    import zipfile
    import os
    from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep, PruningSummaries
    from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_scope, strip_pruning
    from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay as sparsity
    from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude as prune_low_magnitude
    validation_split = 0.2
    prune_low_magnitude = prune_low_magnitude
    #x_train, y_train
    num_images = x_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': sparsity(initial_sparsity=0.20,
                                        final_sparsity=0.90,
                                        begin_step=0,
                                        end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    model_for_pruning.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])

    model_for_pruning.summary()
    logdir = tempfile.mkdtemp()
    print('Writing training logs to ' + logdir)

    callbacks = [UpdatePruningStep(),
                 PruningSummaries(log_dir=logdir, profile_batch=0)
                 ]

    model_for_pruning.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=1,
                     verbose=1,
                     callbacks=callbacks,
                     validation_data=(x_test, y_test))

    model_for_pruning_accuracy = model_for_pruning.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Pruned test accuracy:', model_for_pruning_accuracy)

    model.save('test_pruning6.h5')
    print("save thepruning model")

    import pathlib
    #tf.enable_eager_execution()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("D:/download/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir / "VGG16mnist_model_1_pruning6.tflite"
    tflite_model_file.write_bytes(tflite_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    mnist_train = np.load('D:/download/X5.npy')
    images = tf.cast(mnist_train, tf.float32) / 255.0
    mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)

    def representative_data_gen():
        for input_value in mnist_ds.take(5):
            # Model has only one input so each data point has one element.
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()
    tflite_model_quant_file = tflite_models_dir / "testVGG16_model_quant_int8_0_pruning6.tflite"
    tflite_model_quant_file.write_bytes(tflite_model_quant)


cnn()