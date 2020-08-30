## Deep Active Lesion Segmention (DALS), Code by Ali Hatamizadeh ( http://web.cs.ucla.edu/~ahatamiz/ )

import tensorflow as tf
import tensorflow.contrib.slim as slim
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.01
regularizer = 0.0008
dropout_rate = 0.3
regularizer = tf.contrib.layers.l2_regularizer(scale=regularizer)
w_init_xavi = tf.contrib.layers.xavier_initializer()
w_init_he = tf.contrib.layers.variance_scaling_initializer()
w_init_zero = tf.contrib.layers.variance_scaling_initializer()

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def bottleneck_layer(x, reg, dilation, kernel_init,is_training):

    x = Batch_Normalization(x, is_training)
    x = tf.layers.conv2d(x,filters=4*growth_k,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu, kernel_regularizer=reg, kernel_initializer= kernel_init, dilation_rate=dilation)
    x = tf.layers.dropout(x, rate=dropout_rate, training=is_training,)
    x = Batch_Normalization(x, is_training)
    x = tf.layers.conv2d(x,filters=4*growth_k,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu, kernel_regularizer=reg, kernel_initializer= kernel_init, dilation_rate=dilation)
    x = tf.layers.dropout(x, rate=dropout_rate, training=is_training,)

    return x

def transition_layer(x, reg, kernel_init,is_training):

    x = Batch_Normalization(x, is_training)
    x = tf.layers.conv2d(x,filters=growth_k,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu, kernel_regularizer=reg, kernel_initializer= kernel_init)
    x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)
    x = tf.layers.average_pooling2d(inputs=x, pool_size=2, strides=2, padding="VALID")

    return x

def dense_block(input_x, nb_layers, reg, dilation, kernel_init,is_training):

    layers_concat = list()
    layers_concat.append(input_x)
    x = bottleneck_layer(input_x, reg, dilation, kernel_init,is_training)
    layers_concat.append(x)
    for i in range(nb_layers - 1):
        x = Concatenation(layers_concat)
        x = bottleneck_layer(x, reg, dilation, kernel_init,is_training)
        layers_concat.append(x)
    x = Concatenation(layers_concat)

    return x


def Batch_Normalization(inputs, training):

  return tf.layers.batch_normalization(inputs=inputs,axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, training=training)


def conv_Norm_Relu(x,f,k,s,is_training,padding_mode="same"):

    x=tf.layers.conv2d(x, kernel_initializer= w_init_zero, filters=f, kernel_size=k, strides=s, padding=padding_mode, activation = None, kernel_regularizer=regularizer)

    return tf.nn.relu(Batch_Normalization(x, is_training))

@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, depth=512):
    image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
    image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1", activation_fn=None)
    at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)
    at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)
    at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)
    at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)
    net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,name="concat")
    net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
    
    return net

def transpose_conv_block(x,y,f,k,s,is_training):

    conv1 = tf.concat([tf.layers.conv2d_transpose(x,kernel_initializer= w_init_zero, filters=f, kernel_size=2, strides=2, padding="same", activation = tf.nn.relu, kernel_regularizer=regularizer), y], 3)
    conv1 = tf.layers.conv2d(conv1, kernel_initializer= w_init_zero, filters=f, kernel_size=k, strides=s, padding="same", activation = tf.nn.relu, kernel_regularizer=regularizer)
    conv1 = Batch_Normalization(conv1,is_training)
    conv1 = tf.layers.conv2d(conv1, kernel_initializer= w_init_zero, filters=f, kernel_size=k, strides=s, padding="same", activation = tf.nn.relu, kernel_regularizer=regularizer)
    conv1 = Batch_Normalization(conv1,is_training)

    return conv1


def res_block(x,f,is_training):

    conv1=conv_Norm_Relu(x,f,3,1,is_training,padding_mode="same")
    conv1 = Batch_Normalization(tf.layers.conv2d(conv1,filters=f,kernel_size=3,strides=1,padding="same",activation=None, kernel_regularizer=regularizer, kernel_initializer= w_init_zero))

    return tf.nn.relu(tf.add(conv1,x))


def max_pool(x, p, s):

    pool1 = tf.layers.max_pooling2d(x, pool_size=p, strides=s, padding="valid")

    return pool1


def conv_block(x,f,k,s):

    conv1=conv_Norm_Relu(conv_Norm_Relu(x,f,k,s,padding_mode="same"),f,k,s,padding_mode="same",)

    return tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding="valid")

def conv_start_ddunet(x,f,k,s,is_training):

    conv1 = Batch_Normalization(x, is_training)
    conv1 = tf.layers.conv2d(conv1, kernel_initializer= w_init_zero, filters=f, kernel_size=k, strides=s, padding="same", activation = tf.nn.relu, kernel_regularizer=regularizer)
    return conv1


def upsample_block(x,y):

    size = tf.shape(y)
    up = tf.concat([tf.image.resize_bilinear(x, (size[1], size[2]),align_corners=True), y], 3)

    return up

def dilation_block(x, d1, d2, d3, d4, filter_1,is_training):

    dense51 = tf.layers.conv2d(inputs=x, filters=filter_1,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu, dilation_rate=d1, kernel_regularizer=regularizer, kernel_initializer= w_init_zero)
    dense51 = tf.layers.batch_normalization(inputs=dense51,axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,training=is_training,)
    dense52 = tf.layers.conv2d(inputs=x, filters=filter_1,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu, dilation_rate=d2, kernel_regularizer=regularizer, kernel_initializer= w_init_zero)
    dense52 = tf.layers.batch_normalization(inputs=dense52,axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,training=is_training,)
    dense53 = tf.layers.conv2d(inputs=x, filters=filter_1,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu, dilation_rate=d3, kernel_regularizer=regularizer, kernel_initializer= w_init_zero)
    dense53 = tf.layers.batch_normalization(inputs=dense53,axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,training=is_training,)
    dense54 = tf.layers.conv2d(inputs=x, filters=filter_1,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu, dilation_rate=d4, kernel_regularizer=regularizer, kernel_initializer= w_init_zero)
    dense54 = tf.layers.batch_normalization(inputs=dense54,axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,training=is_training,)
    dilation = tf.concat([dense51, dense52, dense53, dense54],3)

    return dilation


def unet(x):

    conv1=conv_block(x,64,3,1)
    conv2=conv_block(conv1,128,3,1)
    conv3=conv_block(conv2,256,3,1)
    conv4=conv_block(conv3,512,3,1)
    conv5=conv_block(conv4,1024,3,1)
    conv6=upsample_block(conv5,conv4)
    conv7=upsample_block(conv6,conv3)
    conv8=upsample_block(conv7,conv2)
    conv9=upsample_block(conv8,conv1)
    conv10=upsample_block(conv9,conv_Norm_Relu(conv_Norm_Relu(x,256,3,1),256,3,1))
    out_seg=tf.nn.sigmoid(conv_Norm_Relu(conv_Norm_Relu(conv10,256,3,1,padding_mode="same"),1,1,1))

    return out_seg

growth_k = 6

def ddunet(x,is_training):

    conv1 = conv_start_ddunet(x, 64, 7, 2,is_training)
    pool = max_pool(conv1, 2, 2)
    dense1 = dense_block(pool, 3, regularizer,1, w_init_zero,is_training)
    transition1 = transition_layer(dense1, regularizer, w_init_zero,is_training)
    dense2 = dense_block(transition1, 6, regularizer,1, w_init_zero,is_training)
    transition2 = transition_layer(dense2, regularizer, w_init_zero,is_training)
    dense3 = dense_block(transition2, 18, regularizer,1, w_init_zero,is_training)
    transition3 = transition_layer(dense3, regularizer, w_init_zero,is_training)
    dense4 = dense_block(transition3, 12, regularizer,1, w_init_zero,is_training)
    dilation = dilation_block(dense4, 6, 12, 18, 24, 1024,is_training)
    up6 = transpose_conv_block(dilation,dense3,512,3,1,is_training)
    up7 = transpose_conv_block(up6,dense2,256,3,1,is_training)
    up8 = transpose_conv_block(up7,dense1,128,3,1,is_training)
    up9 = transpose_conv_block(up8,conv1,64,3,1,is_training)
    up10 = tf.layers.conv2d_transpose(up9, filters=64,kernel_size=2,strides=2,padding="same", activation = tf.nn.relu, kernel_regularizer=regularizer, kernel_initializer= w_init_zero)
    conv2 = Batch_Normalization(up10, is_training)
    conv11 = tf.layers.conv2d(conv2,filters=128,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu, kernel_regularizer=regularizer, kernel_initializer= w_init_zero)
    conv11 = Batch_Normalization(conv11, is_training)
    out_seg = tf.layers.conv2d(conv11,filters=1,kernel_size=1,strides=1,activation=tf.nn.sigmoid)
    map_lambda1 = tf.exp(tf.divide(tf.subtract(2.0,out_seg),tf.add(1.0,out_seg)))
    map_lambda2 = tf.exp(tf.divide(tf.add(1.0, out_seg), tf.subtract(2.0, out_seg)))

    return out_seg,map_lambda1,map_lambda2







