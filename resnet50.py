from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import ResNet50



train_gen = image_dataset_from_directory(
    directory="Dataset\\Training\\",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(256, 256),
    batch_size=64,
    shuffle=True,
    crop_to_aspect_ratio=False
)

val_gen = image_dataset_from_directory(
    directory="Dataset\\Validation\\",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(256, 256),
    batch_size=64,
    shuffle=True,
    crop_to_aspect_ratio=False
)

test_gen = image_dataset_from_directory(
    directory="Dataset\\Test\\",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(256, 256),
    batch_size=64,
    crop_to_aspect_ratio=False
)




# Carica il modello ResNet50 senza i pesi pre-addestrati
backbone = ResNet50(weights=None, include_top=False, input_shape=(256, 256, 3))


model = Sequential()
model.add(backbone)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary()


my_callbacks = [
    EarlyStopping(monitor="val_loss", patience=3),
    ModelCheckpoint(filepath= 'Models\\ResNet\\V2\\netV2.{epoch:02d}-{val_loss:.2f}.h5'),
    ]


model.fit(train_gen, epochs=20, batch_size=128, validation_data=val_gen, shuffle=True, callbacks=my_callbacks)


history = model.evaluate(test_gen)
print(history)

model.save("Models\\ResNet\\V1\\netV2.h5")
