from models.base_model import ClassificationModel
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

class StochasticDepth(layers.Layer):
    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config

class LayerScale(layers.Layer):
    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"init_values": self.init_values, "projection_dim": self.projection_dim})
        return config
        
class next_time_model(ClassificationModel):
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape, epoch=30, batch_size=32, verbose=1):
        super(next_time_model, self).__init__()
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        self.epoch = epoch 
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = build_model((self.sampling_frequency*10,12),self.n_classes)
        

    def fit(self, X_train, y_train, X_val, y_val):
        callback = tf.keras.callbacks.ModelCheckpoint("./best_model.weights.h5", monitor="val_ROC",
        save_best_only=True, save_weights_only=True, mode="max", save_freq="epoch")
        self.model.fit(X_train, y_train, epochs=self.epoch, batch_size=self.batch_size, 
        validation_data=(X_val, y_val), callbacks = [callback],verbose=self.verbose)
        self.model.load_weights("./best_model.weights.h5")
    def predict(self, X):
        return self.model.predict(X)



def ConvNeXtBlock(projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, name=None):
    def apply(inputs):
        x = inputs
        x = layers.Conv1D(
            filters=projection_dim,
            kernel_size=7,
            padding="same",
            groups=projection_dim,
            name=name + "_depthwise_conv",
        )(x)
        x = layers.LayerNormalization(epsilon=1e-6, name=name + "_layernorm")(x)
        x = layers.Dense(projection_dim * 4, name=name + "_pointwise_conv_1")(x)
        #x = layers.Conv1D(filters=projection_dim,kernel_size=1, padding="same",name=name + "_pointwise_conv_1")(x)
        x = layers.Activation("gelu", name=name + "_gelu")(x)
        x = layers.Dense(projection_dim, name=name + "_pointwise_conv_2")(x)
        #x = layers.Conv1D(filters=projection_dim,kernel_size=1, padding="same",name=name + "_pointwise_conv_2")(x)

        if layer_scale_init_value is not None:
            x = LayerScale(layer_scale_init_value, projection_dim, name=name + "_layer_scale")(x)
        if drop_path_rate:
            x = StochasticDepth(drop_path_rate, name=name + "_stochastic_depth")(x)
        return inputs + x

    return apply

def Head(num_classes=1000, classifier_activation=None, name=None):
    def apply(x):
        x = layers.GlobalAveragePooling1D(name=name + "_head_gap")(x)
        x = layers.LayerNormalization(epsilon=1e-6, name=name + "_head_layernorm")(x)
        x = layers.Dense(num_classes, activation=classifier_activation, name=name + "_head_dense")(x)
        return x

    return apply

def ConvNeXt1D(depths, projection_dims, drop_path_rate=0.0, layer_scale_init_value=6.6484e-06,
               input_shape=None, num_classes=71, classifier_activation="sigmoid"):
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Stem block
    x = layers.Conv1D(projection_dims[0], kernel_size=4, strides=4, name="stem_conv")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="stem_layernorm")(x)
    
    # Stochastic depth schedule
    depth_drop_rates = [float(x) for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
    
    # Downsampling and ConvNeXt stages
    cur = 0
    for i, depth in enumerate(depths):
        if i > 0:
            x = layers.LayerNormalization(epsilon=1e-6, name=f"downsampling_layernorm_{i}")(x)
            x = layers.Conv1D(projection_dims[i], kernel_size=2, strides=2, name=f"downsampling_conv_{i}")(x)
        for j in range(depth):
            x = ConvNeXtBlock(projection_dim=projection_dims[i], 
                              drop_path_rate=depth_drop_rates[cur + j], 
                              layer_scale_init_value=layer_scale_init_value, 
                              name=f"stage_{i}_block_{j}")(x)
        cur += depth

    x = Head(num_classes=num_classes, classifier_activation=classifier_activation, name="head")(x)

    return Model(inputs, x, name="convnext_1d")

def build_model(input_shape, nb_classes, lr_init = 0.001, drop_path_rate=0.15):
    model = ConvNeXt1D(
        depths=[3, 3, 9, 3], 
        projection_dims=[96, 192, 384, 768], 
        drop_path_rate=drop_path_rate, 
        input_shape=input_shape,  
        num_classes=nb_classes
    )
    
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=lr_init), 
                    metrics=[tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.AUC(
                            num_thresholds=200,
                            curve='ROC',
                            summation_method='interpolation',
                            name="ROC",
                            multi_label=True,
                            ),
                        tf.keras.metrics.AUC(
                            num_thresholds=200,
                            curve='PR',
                            summation_method='interpolation',
                            name="PRC",
                            multi_label=True,
                            )],
              )
    print("Inception model built.")
    return model

def scheduler(epoch, lr):
    if epoch % 5 == 0:
        return lr*0.1
    else:
        return lr
