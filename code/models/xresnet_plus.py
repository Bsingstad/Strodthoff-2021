from models.base_model import ClassificationModel
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import ModelCheckpoint


        
class xresnet_plus_model(ClassificationModel):
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape, epoch=50, batch_size=32, lr_init = 0.001, lr_red="yes", model_depth=6, loss="bce", kernel_size=40, bottleneck_size=32, nb_filters=32, clf="binary", verbose=1):
        super(xresnet_plus_model, self).__init__()
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        self.epoch = epoch 
        self.batch_size = batch_size
        self.lr_red = lr_red
        self.verbose = verbose
        self.model = build_xresnet101((self.sampling_frequency*10,12),self.n_classes)
        self.model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(label_smoothing=0.1), optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001), 
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
                    )
          ])

    def fit(self, X_train, y_train, X_val, y_val):
        checkpoint = ModelCheckpoint("model.weights.h5", monitor='val_ROC', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
        self.model.fit(X_train, y_train, epochs=self.epoch, batch_size=self.batch_size, 
            validation_data=(X_val, y_val), verbose=1, callbacks=[checkpoint], shuffle=True)
    def predict(self, X):
        return self.model.predict(X)
    

def build_xresnet101(input_shape, num_classes):
    """Build xresnet101 model for 1D ECG signal classification."""
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv1D(64, 9, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = Activation('gelu')(x)
    #x = GELU()(x)
    #x = layers.MaxPool1D(3, strides=2, padding='same')(x)

    # Residual stages
    # Stage 1 (3 blocks)
    for _ in range(3):
        x = residual_block(x, 64)
    
    # Stage 2 (4 blocks)
    x = residual_block(x, 128, stride=2)
    for _ in range(3):
        x = residual_block(x, 128)
    
    # Stage 3 (23 blocks)
    x = residual_block(x, 256, stride=2)
    for _ in range(22):
        x = residual_block(x, 256)
    
    # Stage 4 (3 blocks)
    x = residual_block(x, 512, stride=2)
    for _ in range(2):
        x = residual_block(x, 512)
    
    # Final layers
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    return Model(inputs, outputs)


def se_block(input_tensor, reduction=16):
    """Squeeze-and-Excitation block."""
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Reshape((1, filters))(se)
    se = layers.Dense(filters // reduction, activation='gelu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.multiply([input_tensor, se])


def residual_block(x, filters, stride=1):
    """Basic residual block with two 3x1 conv layers and identity shortcut."""
    shortcut = x
    
    # Check if we need to adjust the shortcut connection
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # First convolution
    x = layers.Conv1D(filters, 5, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = Activation('gelu')(x)
    #x = GELU()(x)
    
    # Second convolution
    x = layers.Conv1D(filters, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = se_block(x)
    
    # Add shortcut and activate
    x = layers.Add()([x, shortcut])
    x = Activation('gelu')(x)
    #x = GELU()(x)
    
    return x

