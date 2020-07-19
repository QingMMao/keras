from keras import layers
from keras import models


model = models.Sequential()
model.add(layers.Dense(32,activation = 'relu',input_shape = (784,)))
model.add(layers.Dense(10,activation = 'softmax'))

input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = models.Model(input=input_tensor,output=output_tensor)


from keras import optimizers

model.compile(optimizer = optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['accuracy'])
#There is no target_tensor!!!
model.fit(input_tensor,target_tensor,batch_size=128,epochs=10)

