import tensorflow as tf


layers = tf.keras.layers


class PCEN(tf.keras.layers.Layer):

    def __init__(self):
        super(PCEN, self).__init__()

        self.alpha_init = tf.random_normal_initializer(mean=1.0, stddev=0.1)
        power_init = tf.random_normal_initializer(mean=1.0, stddev=0.1)
        bias_init = tf.random_normal_initializer(mean=1.0, stddev=0.1)

        # 0.015 and 0.08 may help
        self.smooth_coef = tf.constant(0.015)
        self.eps = tf.constant(1e-9)

        self.alpha = tf.Variable(initial_value=self.alpha_init(shape=(1,)), trainable=True, name='alpha_rni')
        self.power = tf.Variable(initial_value=power_init(shape=(1,)), trainable=True, name='power_rni')
        self.bias = tf.Variable(initial_value=bias_init(shape=(1,)), trainable=True, name='bias_rni')

    def call(self, inputs):
        time_steps = inputs.shape[2]
        M_prev = tf.multiply(self.smooth_coef, inputs[:, :, 0:1, :])
        before = M_prev

        for element in range(time_steps - 1):
            element += 1

            M = tf.multiply((1 - self.smooth_coef), M_prev) + tf.multiply(self.smooth_coef,
                                                                          inputs[:, :, element:element + 1, :])

            before = tf.concat([before, M], axis=2)

            M_prev = M
        M = before
        return tf.pow((inputs / tf.pow((self.eps + M), self.alpha) + self.bias), self.power) - tf.pow(self.bias,
                                                                                                      self.power)

class _Identity_block(tf.keras.Model):

    def __init__(self, kernel_size, filters, stage, block):
        super(_Identity_block, self).__init__()
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filters

        # step1
        self.conv2a = layers.Conv2D(F1, (1, 1),
                                    name=conv_name_base + '2a')
        self.bn2a = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')

        # step2
        self.conv2b = layers.Conv2D(F2, kernel_size, padding='same',
                                    name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')

        # step3
        self.conv2c = layers.Conv2D(F3, (1, 1),
                                    name=conv_name_base + '2c')
        self.bn2c = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')

    def call(self, input_array, training=False):
        x = self.conv2a(input_array)
        x = self.bn2a(x, training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training)

        x += input_array
        x = tf.nn.relu(x)

        return x


class _Conv_block(tf.keras.Model):

    def __init__(self, kernel_size, filters, stage, block, strides):
        super(_Conv_block, self).__init__()
        conv_base_name = 'res' + str(stage) + block + '_branch'
        bn_base_name = 'bn' + str(stage) + block + '_branch'

        f1, f2, f3 = filters

        # step1
        self.conv2a = layers.Conv2D(f1, (1, 1),
                                    strides=strides, name=conv_base_name + '2a')
        self.bn2a = layers.BatchNormalization(axis=3, name=bn_base_name + '2a')

        # step2
        self.conv2b = layers.Conv2D(f2, kernel_size, padding='same',
                                    strides=(1, 1), name=conv_base_name + '2b')
        self.bn2b = layers.BatchNormalization(axis=3, name=bn_base_name + '2b')

        # step3
        self.conv2c = layers.Conv2D(f3, (1, 1),
                                    strides=(1, 1), name=conv_base_name + '2c')
        self.bn2c = layers.BatchNormalization(axis=3, name=bn_base_name + '2c')

        # shorcut
        self.conv_shortcut = layers.Conv2D(f3, (1, 1),
                                           strides=strides, name=conv_base_name + '1')
        self.bn_shortcut = layers.BatchNormalization(axis=3, name=bn_base_name + '1')

    def call(self, input_array, training=False):
        x = self.conv2a(input_array)
        x = self.bn2a(x, training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training)

        shortcut = self.conv_shortcut(input_array)
        shortcut = self.bn_shortcut(shortcut, training)

        x += shortcut
        x = tf.nn.relu(x)

        return x


class Resnet50(tf.keras.Model):

    def __init__(self, classes, trainable=True):

        super(Resnet50, self).__init__()

        def Identity_block(filters, stage, block):
            return _Identity_block(3, filters, stage, block)

        def Conv_Block(filters, stage, block, strides=(2, 2)):
            return _Conv_block(3, filters, stage, block, strides)

        #def PCEN_Layer():
        #    return PCEN()

        #self.bg_Layer = PCEN_Layer()
        self.zeroPadding = layers.ZeroPadding2D((3, 3), name='ZeroPad_Intro')

        # We can use 'valid' or 'same' padding in the first Conv, doesn't seems to
        # Intro 1
        self.conv1 = layers.Conv2D(64, (7, 7),
                                   strides=(2, 2), padding='same', name='Conv_Intro')
        self.bn1 = layers.BatchNormalization(axis=3, name='Bn_Intro')
        self.max1 = layers.MaxPooling2D((3, 3), strides=(2, 2), name='Max_pool')

        # Collections 2
        self.s2a = Conv_Block([64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.s2b = Identity_block([64, 64, 256], stage=2, block='b')
        self.s2c = Identity_block([64, 64, 256], stage=2, block='c')

        # Collections 3
        self.s3a = Conv_Block([128, 128, 512], stage=3, block='a')
        self.s3b = Identity_block([128, 128, 512], stage=3, block='b')
        self.s3c = Identity_block([128, 128, 512], stage=3, block='c')
        self.s3d = Identity_block([128, 128, 512], stage=3, block='d')

        # Collections 4
        self.s4a = Conv_Block([256, 256, 1024], stage=4, block='a')
        self.s4b = Identity_block([256, 256, 1024], stage=4, block='b')
        self.s4c = Identity_block([256, 256, 1024], stage=4, block='c')
        self.s4d = Identity_block([256, 256, 1024], stage=4, block='d')
        self.s4e = Identity_block([256, 256, 1024], stage=4, block='e')
        self.s4f = Identity_block([256, 256, 1024], stage=4, block='f')

        # Collections 5
        self.s5a = Conv_Block([512, 512, 2048], stage=5, block='a')
        self.s5b = Identity_block([512, 512, 2048], stage=5, block='b')
        self.s5c = Identity_block([512, 512, 2048], stage=5, block='c')

        self.avg_pool = layers.AveragePooling2D(pool_size=(2, 2))
        # Why is AvPool7 not working?
        # self.avg_pool = layers.AveragePooling2D(pool_size=(7, 7), strides=(7,7))

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(classes, activation='softmax', name='dense_Outputs')

    def call(self, inputs, training=True):

        #x_hat = self.bg_Layer(inputs)

        x = self.zeroPadding(inputs)

        # -1
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = tf.nn.relu(x)
        x = self.max1(x)

        # -2
        x = self.s2a(x, training)
        x = self.s2b(x, training)
        x = self.s2c(x, training)

        # -3
        x = self.s3a(x, training)
        x = self.s3b(x, training)
        x = self.s3c(x, training)
        x = self.s3d(x, training)

        # -4
        x = self.s4a(x, training)
        x = self.s4b(x, training)
        x = self.s4c(x, training)
        x = self.s4d(x, training)
        x = self.s4e(x, training)
        x = self.s4f(x, training)

        # -5
        x = self.s5a(x, training)
        x = self.s5b(x, training)
        x = self.s5c(x, training)

        x = self.avg_pool(x)

        # -fullyConnected layer
        x = self.flatten(x)
        x = self.dense(x)

        return x

