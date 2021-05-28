import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image



train_datagen =ImageDataGenerator(rescale=1./255)
valid_datagen=ImageDataGenerator(rescale=1./255)

train_dir="dogsandcats/train/"
test_dir="dogsandcats/test/"

train_data = train_datagen.flow_from_directory(train_dir,batch_size=32,
                                               target_size=(64,64),
                                               class_mode="binary",
                                               seed=42,
                                               )

valid_data = train_datagen.flow_from_directory(test_dir,batch_size=32,
                                               target_size=(64,64),
                                               class_mode="binary",
                                               seed=42,
                                               )









# create the model
model_1 = Sequential([
  Conv2D(filters=10,
         kernel_size=3, # can also be (3, 3)
         activation="relu",
         input_shape=(64, 64, 3)), # first layer specifies input shape (height, width, colour channels)
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
         padding="valid"), # padding can also be 'same'
  Conv2D(10, 3, activation="relu"),
  Conv2D(10, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
  MaxPool2D(2),
  Flatten(),
  Dense(1, activation="sigmoid") # binary activation output
])



model_1.compile(loss="binary_crossentropy",
               optimizer=tensorflow.keras.optimizers.Adam(),
               metrics=["accuracy"])

history_1= model_1.fit(train_data,
                      epochs=5,
                      steps_per_epoch=len(train_data),
                      validation_data=valid_data,
                      validation_steps=len(valid_data))

model_1.save("model.h5")
print("Saved model to disk")

def load_and_prep_image(filename, img_shape=64):
    # Read in the image
    img = tensorflow.io.read_file(filename)
    # Decode it into a tensor
    img = tensorflow.image.decode_jpeg(img)
    # Resize the image
    img = tensorflow.image.resize(img, [64, 64])
    # Rescale the image (get all values between 0 and 1)
    img = img / 255.
    return img


animal = load_and_prep_image("cat1.jpg")

animal = tensorflow.expand_dims(animal, axis=0)

pred = model_1.predict(animal)


train_data.class_indices
if pred[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)