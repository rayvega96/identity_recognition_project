from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping




train_gen, val_gen = image_dataset_from_directory(
    directory="Dataset\\Resized\\",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(256, 256),
    batch_size=64,
    validation_split=0.4,
    subset="both",
    seed=64,
    shuffle=True,
    crop_to_aspect_ratio=True
)

test_gen = image_dataset_from_directory(
        directory="Dataset\\Test\\",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(256, 256),
    batch_size=64,
    crop_to_aspect_ratio=True
)

model = Sequential()
model.add(Conv2D(32, (7, 7), padding='same', activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics="categorical_accuracy")
model.summary()

input("Premi un tasto per iniziare l'addestramento.")

callbacks = [ EarlyStopping(monitor="val_loss", patience=3), ModelCheckpoint(filepath="Models\\CustomNet\\V4\\net-{epoch:02d}-{val_loss:.2f}.h5")

]

model.fit(train_gen, batch_size=128, epochs=20, validation_data=val_gen, shuffle=True, callbacks=callbacks)

history = model.evaluate(test_gen)
print(history)

model.save("Models\\CustomNet\\V4\\netV4.h5")

