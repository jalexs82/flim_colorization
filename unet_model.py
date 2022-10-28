import os
import logging
logging.disable(logging.WARNING)
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, SpatialDropout2D


def dense_unet(input_shape, n_classes, n_levels=5, start_n_filters=64, last_activation='softmax', dropout_type=None,
               sp_dim_rate=0.3):
    inputs = Input(input_shape)
    feed_layer = inputs
    pass_layers = []
    for layer in range(n_levels - 1):
        n_filters, dropouts = get_layer_props(layer, start_n_filters)
        down_conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(feed_layer)
        down_conv1 = dropout_features(down_conv1, dropouts, dropout_type, sp_dim_rate)
        down_concat1 = concatenate([feed_layer, down_conv1])
        down_conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(down_concat1)
        down_concat2 = concatenate([feed_layer, down_conv2])
        feed_layer = MaxPooling2D(pool_size=(2, 2))(down_concat2)
        if layer == n_levels - 2:
            pass_layers.append(down_concat2)
        else:
            pass_layers.append(down_conv2)

    n_filters, dropouts = get_layer_props(n_levels - 1, start_n_filters)
    bot_conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(feed_layer)
    bot_conv1 = dropout_features(bot_conv1, dropouts, dropout_type, sp_dim_rate)
    bot_concat1 = concatenate([feed_layer, bot_conv1])
    bot_conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(bot_concat1)
    feed_layer = concatenate([feed_layer, bot_conv2])

    for layer in reversed(range(n_levels - 1)):
        n_filters, dropouts = get_layer_props(layer, start_n_filters)
        up_convT = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(feed_layer)
        up_concat1 = concatenate([up_convT, pass_layers[layer]])
        up_conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up_concat1)
        up_conv1 = dropout_features(up_conv1, dropouts, dropout_type, sp_dim_rate)
        up_concat2 = concatenate([up_concat1, up_conv1])
        up_conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up_concat2)
        feed_layer = concatenate([up_concat1, up_conv2])

    final_conv = Conv2D(n_classes, (1, 1), activation=last_activation)(feed_layer)

    model = Model(inputs=[inputs], outputs=[final_conv])

    return model


def get_layer_props(layer_num, start_n_filters):
    n_filters = start_n_filters * (2 ** layer_num)
    dropouts = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3][layer_num]

    return n_filters, dropouts


def dropout_features(input_layer, dropouts, dropout_type, sp_dim_rate):
    if dropout_type == 'normal':
        output_layer = Dropout(dropouts)(input_layer)
        return output_layer
    elif dropout_type == 'spatial':
        output_layer = SpatialDropout2D(sp_dim_rate)(input_layer)
        return output_layer
    else:
        return input_layer
