
base_dir = './dataset'

train_dir = f'{base_dir}/train'

val_dir = f'{base_dir}/val'

test_dir = f'{base_dir}/test'

batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(

    train_dir,

    label_mode='categorical',

    image_size=(224, 224),

    batch_size=batch_size,

    shuffle=True

)

val_ds = tf.keras.utils.image_dataset_from_directory(

    val_dir,

    label_mode='categorical',

    image_size=(224, 224),

    batch_size=batch_size,

    shuffle=True

)

test_ds = tf.keras.utils.image_dataset_from_directory(

    test_dir,

    label_mode='categorical',

    image_size=(224, 224),

    batch_size=batch_size,

    shuffle=False

)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=5

)

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test accuracy: {test_accuracy:.2f}")
