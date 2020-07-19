
from keras.models import Sequential, Model
from keras import layers
from keras import Input

seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64, )))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

input_tensor = Input(shape=(64, ))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)

#print(model.summary())#P224

#An error try,this will get a RuntimeError
#unrelated_input = Input(shape=(32, ))
#bad_model = model = Model(unrelated_input, output_tensor)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

import numpy as np
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, epochs=10, batch_size=128)

score = model.evaluate(x_train, y_train)


#Mutiple-input
from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocalbulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtpe='int32', name='question')
embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(answer_vocalbulary_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

import numpy as np

num_samples = 5000
max_length = 10000
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))

answers = np.random.randint(0, 1, size=(num_samples, answer_vocalbulary_size))

model.fit([text, question], answers, epochs10, batch_size=128)
#If they were named
#model.fit({'text':text, 'question':question}, answers,epochs=10,batch_size)

#Mutiple-output

vocabulary_size = 50000
num_income_groups = 10
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])

model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy',
                    'gender': 'binary_crossentropy'})

model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])

model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy',
                    'gender': 'binary_crossentropy'},
              loss_weights={'age': 0.25,
                            'income': 1.,
                            'gender': 10.})

model.fit(posts, [age_targets, income_targets, gender_targets],
          epochs=10, batch_size=64)

model.fit(posts, {'age': age_targets,
                  'income': income_targets,
                  'gender': gender_targets},
          epochs=10, batch_size=64)

#Inception module
from keras import layers
# We assume the existence of a 4D input tensor `x`
# Every branch has the same stride value (2), which is necessary to keep all
# branch outputs the same size, so as to be able to concatenate them.
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
# In this branch, the striding occurs in the spatial convolution layer
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
# In this branch, the striding occurs in the average pooling layer
branch_c = layers.AveragePooling2D(3, strides=2, activation='relu')(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
# Finally, we concatenate the branch outputs to obtain the module output
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

#Implementing a residual connection when feature map sizes are the same: using identity residual connections
x = ...
y = layers.Conv2D(128, 3, activation='relu')(x)
y = layers.Conv2D(128, 3, activation='relu')(y)
y = layers.Conv2D(128, 3, activation='relu')(y)

y = layers.add([y, x])
#Implementing a residual connection when feature map sizes differ: using a linear residual connection
# We assume the existence of a 4D input tensor `x`
x = ...
y = layers.Conv2D(128, 3, activation='relu')(x)
y = layers.Conv2D(128, 3, activation='relu')(y)
y = layers.MaxPooling2D(2, strides=2)(y)
# We use a 1x1 convolution to linearly downsample
# the original `x` tensor to the same shape as `y`
residual = layers.Conv2D(1, strides=2)(x)
# We add the residual tensor back to the output features
y = layers.add([y, residual])

#Layer weight sharing (i.e. layer reuse) with the functional API: implementing a siamese LSTM model
# We instantiate a single LSTM layer, once
lstm = layers.LSTM(32)
# Building the left branch of the model
# -------------------------------------
# Inputs are variable-length sequences of vectors of size 128
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)
# Building the right branch of the model
# --------------------------------------
right_input = Input(shape=(None, 128))
# When we call an existing layer instance,
# we are reusing its weights
right_output = lstm(right_input)
# Building the classifier on top
# ------------------------------
merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)
# Instantiating and training the model
# ------------------------------------
model = Model([left_input, right_input], predictions)
# When you train such a model, the weights of the `lstm` layer
# are updated based on both inputs.
model.fit([left_data, right_data], targets)

#Implementing a siamese vision model (shared convolutional base)
# Our base image processing model with be the Xception network
# (convolutional base only).
xception_base = applications.Xception(weights=None, include_top=False)
# Our inputs are 250x250 RGB images.
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))
# We call the same vision model twice!
left_features = xception_base(left_input)
right_input = xception_base(right_input)
# The merged features contain information from both
# the right visual feed and the left visual feed
merged_features = layers.concatenate([left_features, right_input], axis=-1)


#EarlyStopping and ModelCheckpoint
import keras

callbacks_list = [
        keras.callbacks.EarlyStopping(
                monitor = 'acc',
                patience = 1,
                ),
        keras.callbacks.ModelCheckpoint(
                filepath = 'my_model.h5',
                monitor = 'val_loss',
                save_best_only = True,
                )
        ]
from keras.models import Sequential

model = Sequential()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))
#Another
callbacks_list = [
keras.callbacks.ReduceLROnPlateau(
# This callback will monitor the validation loss of the model
monitor='val_loss',
# It will divide the learning by 10 when it gets triggered
factor=0.1,
# It will get triggered after the validation loss has stopped improving
# for at least 10 epochs
patience=10,
)
]
# Note that since the callback will be monitor validation loss,
# we need to pass some `validation_data` to our call to `fit`.
model.fit(x, y,
epochs=10,
batch_size=32,
callbacks=callbacks_list,
validation_data=(x_val, y_val))



import numpy as np

class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer_output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outputs)
        
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requries validation_data.')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()
































