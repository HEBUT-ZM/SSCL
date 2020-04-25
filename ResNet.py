import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from sklearn.cluster import k_means
NUM_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
FC_WEIGHT_STDDEV = 0.01

class ResNetModel(object):
    def __init__(self, images, u_image, labels, is_training, depth, num_classes, BATCHSIZE):      
        self.image=images
        self.u_image=u_image
        self.images = self._get_concats()
        self.labels = labels
        self.is_training = is_training
        self.num_classes = num_classes
        self.depth = depth
        self.BATCHSIZE = BATCHSIZE
        self.embeddings = self._get_embeddings()
        self.pred_prob, self.softmax_loss = self._get_regular_loss()
        self.centers_update_op, self.centers=self._get_center_loss()       
        self.predictions = self._get_pred()
        self.accuracy = self._get_accuracy()
        self.num_blocks = NUM_BLOCKS[depth]
    def _get_concats(self):
        cimages=tf.concat([self.image,self.u_image],0)
        return cimages
    def _get_embeddings(self):
        return self.inference()
    def _get_regular_loss(self):
        return self.Regular_Softmax_Loss( self.embeddings, self.labels, self.BATCHSIZE)
    def _get_center_loss(self):
        return self.Center_Loss(self.embeddings, self.labels, self.num_classes, self.BATCHSIZE)
    def _get_pred(self):
        return tf.argmax(self.pred_prob, axis=1)
    def _get_accuracy(self):
        correct_predictions = tf.equal(self.predictions, self.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
        return accuracy
    def inference(self):
        with tf.variable_scope('scale1'):
            s1_conv = conv(self.images, ksize=7, stride=2, filters_out=64)
            s1_bn = bn(s1_conv, self.is_training)
            s1 = tf.nn.relu(s1_bn)
        with tf.variable_scope('scale2'):
            s2_mp = tf.nn.max_pool(s1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            s2 = stack(s2_mp, self.is_training, num_blocks=3, stack_stride=1, block_filters_internal=64)
        with tf.variable_scope('scale3'):
            s3 = stack(s2, self.is_training, num_blocks=4, stack_stride=2, block_filters_internal=128)
        with tf.variable_scope('scale4'):
            s4 = stack(s3, self.is_training, num_blocks=6, stack_stride=2, block_filters_internal=256)
        with tf.variable_scope('scale5'):
            s5 = stack(s4, self.is_training, num_blocks=3, stack_stride=2, block_filters_internal=512)
        avg_pool = tf.reduce_mean(s5, reduction_indices=[1, 2], name='avg_pool')
        return avg_pool
    def Regular_Softmax_Loss(self, embeddings, labels, BATCHSIZE):
        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[embeddings.get_shape().as_list()[-1], self.num_classes],
                                      initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
            biases = _get_variable('biases', shape=[self.num_classes], initializer=tf.zeros_initializer())
            updated_logits=tf.nn.xw_plus_b(embeddings[0:BATCHSIZE,:], weights, biases)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=updated_logits))
            pred_prob = tf.nn.softmax(logits=updated_logits)                                
            return pred_prob, loss
    def Center_Loss(self, features, labels, num_classes, BATCHSIZE):
        len_features = features.get_shape()[1]
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
        labels = tf.reshape(labels, [-1])
        labels=tf.cast(labels, tf.int32)       
        centers_batch = tf.gather(centers, labels)
        diff = centers_batch - features[0:BATCHSIZE,:]   
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])    
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = 0.01 * diff    
        centers_update_op = tf.scatter_sub(centers, labels, diff)       
        return centers_update_op, centers    
    def optimize(self, loss, learning_rate, train_layers=[]):
        trainable_var_names = ['weights', 'biases', 'beta', 'gamma','embedding_weights']
        var_list = [v for v in tf.trainable_variables() if
            v.name.split(':')[0].split('/')[-1] in trainable_var_names and
            contains(v.name, train_layers)]
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #gradients=optimizer.compute_gradients(loss,var_list)   
        #soft_gradients=optimizer.compute_gradients(loss,var_soft_list)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss]))
        batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)   
        return tf.group(train_op, batchnorm_updates_op)
    def load_original_weights(self, weight_path, session):
        weights_path = weight_path + 'ResNet-L{}.npy'.format(self.depth)
        weights_dict = np.load(weights_path, encoding='bytes', allow_pickle=True).item()
        for op_name in weights_dict:
            parts = op_name.split('/')
            if parts[0] == 'fc' and self.num_classes != 1000:
                continue
            full_name = "{}:0".format(op_name)
            var = [v for v in tf.global_variables() if v.name == full_name][0]
            session.run(var.assign(weights_dict[op_name]))

"""
Helper methods
"""
def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                           trainable=trainable)
def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=shape, dtype='float', initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
def bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))
    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())
    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
def stack(x, is_training, num_blocks, stack_stride, block_filters_internal):
    for n in range(num_blocks):
        block_stride = stack_stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, is_training, block_filters_internal=block_filters_internal, block_stride=block_stride)
    return x
def block(x, is_training, block_filters_internal, block_stride):
    filters_in = x.get_shape()[-1]
    m = 4
    filters_out = m * block_filters_internal
    shortcut = x
    with tf.variable_scope('a'):
        a_conv = conv(x, ksize=1, stride=block_stride, filters_out=block_filters_internal)
        a_bn = bn(a_conv, is_training)
        a = tf.nn.relu(a_bn)
    with tf.variable_scope('b'):
        b_conv = conv(a, ksize=3, stride=1, filters_out=block_filters_internal)
        b_bn = bn(b_conv, is_training)
        b = tf.nn.relu(b_bn)
    with tf.variable_scope('c'):
        c_conv = conv(b, ksize=1, stride=1, filters_out=filters_out)
        c = bn(c_conv, is_training)
    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or block_stride != 1:
            shortcut_conv = conv(x, ksize=1, stride=block_stride, filters_out=filters_out)
            shortcut = bn(shortcut_conv, is_training)
    return tf.nn.relu(c + shortcut)
def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.contrib.layers.xavier_initializer()
    weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    return tf.nn.xw_plus_b(x, weights, biases)
def contains(target_str, search_arr):
    rv = False
    for search_str in search_arr:
        if search_str in target_str:
            rv = True
            break
    return rv