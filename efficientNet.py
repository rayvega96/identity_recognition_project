import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.preprocessing.image import ImageDataGenerator
from keras import layers


# Carica il modello pre-addestrato di EfficientNetB0
backbone = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Imposta i parametri
num_classes = 5  # 4 identità + classe "Unknown"
input_shape = (64, 64, 3)

# Congela i pesi del backbone
backbone.trainable = False

# Aggiungi un livello di classificazione finale
model = tf.keras.Sequential([
    backbone,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax')
])

# Compila il modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Stampa un riassunto del modello
model.summary()

# Imposta le directory dei dati di addestramento e validazione
train_dir = 'training_set'
valid_dir = 'validation_set'

# Crea generatori di immagini per l'addestramento e la validazione
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Specifica la dimensione dell'input
target_size = (64, 64)

# Carica e pre-processa i dati di addestramento e validazione
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical'
)

# Addestra il modello
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)
)


# Salva il modello addestrato
model.save('model/model_name.h5')