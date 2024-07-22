import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf


def make_content_space_classif(n_sample_wiener, feat_wiener, n_labels) -> tf.keras.Model:
    init = tf.keras.initializers.RandomNormal(seed=42)

    _input = tf.keras.Input((n_sample_wiener, feat_wiener))

    x = tf.keras.layers.Conv1D(16, 3, 1, padding='same', kernel_initializer=init)(_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(n_labels)(x)
    
    model = tf.keras.Model(_input, x)


    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy()
    )

    return model


def make_style_space_classifier(style_vector_shape:tuple, n_labels:int):

    _input = tf.keras.Input((style_vector_shape,))

    x = tf.keras.layers.Dense(n_labels)(_input)
    
    model = tf.keras.Model(_input, x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy()
        )

    return model


def make_real_fake_classifier(seq_shape:tuple)-> tf.keras.models.Model:
    # initializer = tf.keras.initializers.GlorotNormal()

    model = tf.keras.Sequential()


    model.add(tf.keras.Input(shape=seq_shape))

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=50, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=tf.keras.metrics.Accuracy()
    )

    return model