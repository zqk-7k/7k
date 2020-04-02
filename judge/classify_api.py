from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

(train_data, validation_data, test_data) = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch

# 定义一个简单的序列模型
def create_model():
    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[],
    dtype=tf.string, trainable=True)
    hub_layer(train_examples_batch[:3])
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def classify(content):
    # 创建一个基本的模型实例
    model = create_model()

    # 显示模型的结构
   # model.summary()

    model.load_weights('C:/Users/56236/Desktop/2.1tensorflow/checkpoints/my_tfcheckpoint')

    text = np.array(['content'])

    #label = np.array([1, 0])
    print('3')
    #result = model.evaluate(text, label)
    y_pred = model.predict(text, batch_size=1)
    return(y_pred)