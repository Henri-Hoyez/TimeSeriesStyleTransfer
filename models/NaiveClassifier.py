import tensorflow as tf

def make_naive_discriminator(seq_shape:tuple, n_classes:int)-> tf.keras.models.Model:
    # initializer = tf.keras.initializers.GlorotNormal()

    model = tf.keras.Sequential()


    model.add(tf.keras.Input(shape=seq_shape))

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    # model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())

    # model.add(tf.keras.layers.Dense(units=100, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=50))
    # model.add(tf.keras.layers.LeakyReLU())
    # model.add(tf.keras.layers.Activation("relu"))

    
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=n_classes))
    # model.add(tf.keras.layers.Dense(units=n_classes))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy()
    )

    return model

