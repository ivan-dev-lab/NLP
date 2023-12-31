import tensorflow as tf

def FullyConnected_Keras (input_shape: tuple):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=((input_shape[0],))))
    model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))

    model.add(tf.keras.layers.Dense(units=input_shape[1], activation='softmax'))

    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model

