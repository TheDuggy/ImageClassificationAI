"""
Copyright 2023 Georg Kollegger

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import os

data_dir = './data/animals_big_translated/'


# In[ ]:


train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224,224),
    batch_size=32
)

validate_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224,224),
    batch_size=32
)

print('train_ds-classes: ' + str(train_dataset.class_names))
print('val_ds-classes: ' + str(validate_dataset.class_names))


# In[ ]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = train_dataset.prefetch(buffer_size=AUTOTUNE)


# In[ ]:


normalization_layer = tf.keras.layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))


# In[ ]:


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='saves/',
                                                 save_weights_only=True,
                                                 verbose=1)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(224, 224, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(train_dataset.class_names))
])

if os.path.exists('./save/'):
  model.load_weights('./save/')

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[cp_callback,]
)

model.save('./export/img_class_ai')

with open('./history.json', 'a') as f:
    f.write(json.dumps(history.history))

with open('./classes.json', 'a') as f:
  f.write(json.dumps({'classes-train-ds: ' : train_dataset.class_names, 'classes-val-ds': validate_dataset.class_names}))


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./training.png')

