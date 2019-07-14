import tensorflow as tf


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    # First conponent
    # X = tf.nn.conv2d(X, F1, strides = (1, 1), padding='valid', name = conv_name_base + '2a')
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               name=conv_name_base + '2a',
                               kernel_initializer=tf.keras.initializers.glorot_uniform())(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # X = tf.nn.conv2D()

    # Second one
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                               name=conv_name_base + '2b',
                               kernel_initializer=tf.keras.initializers.glorot_uniform())(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third one
    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               name=conv_name_base + '2c',
                               kernel_initializer=tf.keras.initializers.glorot_uniform())(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Adding Shortcut to main path
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def conv_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    # First conponent
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                               name=conv_name_base + '2a',
                               kernel_initializer=tf.keras.initializers.glorot_uniform())(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second conponent
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                               name=conv_name_base + '2b',
                               kernel_initializer=tf.keras.initializers.glorot_uniform())(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third conponent
    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               name=conv_name_base + '2c',
                               kernel_initializer=tf.keras.initializers.glorot_uniform())(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Shortcut
    X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                        name=conv_name_base + '1',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform())(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Adding
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def ResNet_50(input_shape=(64, 64, 3), classes=6):
    """
    Creates ResNet architecture with some modifications to match Audio classification architectures
    """

    # Intro
    X_input = tf.keras.layers.Input(input_shape)

    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    # 1
    # Original first convolution
    X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)

    # Adaptation from "CNN arquitectures for audio event detection" paper

    # X = tf.keras.layers.Conv2D(64, (7,7), strides=(1,1), padding='valid',
    #                         kernel_initializer=tf.keras.initializers.glorot_uniform())
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv_1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2))(X)

    # 2
    X = conv_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # 3
    X = conv_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='d')

    # 4
    X = conv_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # 5
    X = conv_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    # Original one
    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # Modified one
    # X = tf.keras.layers.AveragePooling2D(pool_size=(6, 4), name='avg_pool')(X)

    # Output

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes),
                              kernel_initializer=tf.keras.initializers.glorot_uniform())(X)

    # Modeling

    model = tf.keras.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model
