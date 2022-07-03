import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# 输出样例
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_labels_batch
# # 不能下载，将nnlm-en-dim50_2文件夹移动到工作目录下
# embedding = "nnlm-en-dim50_2"
# hub_layer = hub.KerasLayer(embedding, input_shape=[],
#                            dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3])

#embedding = "https://hub.tensorflow.google.cn/google/nnlm-en-dim50/2"
embedding = "nnlm-en-dim50_2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

# 模型编译
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(1000).batch(52),
                    epochs=10,
                    validation_data=validation_data.batch(52),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

