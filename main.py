import time
import tensorflow as tf
import numpy
import os
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import convert_to_tensor


import muctface_input
import celebaface_input

from recognizor import EncodeNet, FeatureNet

from tensorflow.contrib.keras.python.keras.activations import softmax
from tensorflow.contrib.learn.python.learn.trainable import Trainable

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True


record_log = True
enet_log_dir = './log/enet_muct_16/'
fnet_log_dir = './log/fnet_celeba/'
embedding_log_dir = './log/embedding_log_dir/'


iters = int(-1)

is_training = tf.placeholder(tf.bool, name='ph_is_training')
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(0.001, global_step, 500, 1, staircase=False, name='learning_rate')


# ========================

[image64_muct, label64_muct] = muctface_input.inputs(False, './muct-faces/', 64)  # -1.~1.
image64_muct = image64_muct / 127.5 - 1.0
label64_muct = tf.reshape(tf.cast(label64_muct, tf.int32), [-1])

image64_celeba = celebaface_input.inputs('./celeba-faces/', 64)  # -1.~1.
image64_celeba = image64_celeba / 127.5 - 1.0




with tf.variable_scope('fnet'):
    fnet_celeba = FeatureNet(image64_celeba, is_training=is_training)
with tf.variable_scope('fnet', reuse=True):    
    fnet_muct = FeatureNet(image64_muct, is_training=is_training)

with tf.variable_scope('enet'):
    enet = EncodeNet(fnet_muct.feature_output, 16, muctface_input.CLASS_NUMS, is_training=is_training)


with tf.variable_scope('loss'):
    fnet_loss = tf.reduce_mean(tf.square(fnet_celeba.olayer4.output - image64_celeba)) * 127.5
    
    weight_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label64_muct, logits=enet.layer2.logit))
    accuarcy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(enet.layer2.logit, tf.reshape(label64_muct, [-1]) , 1), tf.float32))
    
    centers = tf.get_variable('centers', [enet.layer2.output_size, enet.layer1.output_size], dtype=tf.float32,
            initializer=tf.zeros_initializer(tf.float32), trainable=False)
    center_distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(centers), axis=1)))
    
    centers_batch = tf.gather(centers, label64_muct)
    center_loss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(enet.layer1.output - centers_batch), axis=1)))

# ===================

fnet_train = tf.train.AdamOptimizer(learning_rate).minimize(
    fnet_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fnet") , global_step=global_step)
enet_train = tf.train.AdamOptimizer(learning_rate).minimize(
        softmax_loss + center_loss + weight_loss, var_list=[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="enet"), centers ], global_step=global_step)

# ===================
enet_merged = tf.summary.merge([tf.summary.scalar('softmax_loss', softmax_loss), tf.summary.scalar('center_loss', center_loss),
                                tf.summary.scalar('center_distance', center_distance)])
fnet_merged = tf.summary.merge([tf.summary.scalar('fnet_loss', fnet_loss), tf.summary.image('fnet_restore_image', (fnet_celeba.olayer4.output + 1) / 2, 64)])


sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()



# ========restore===============
fnet_saver = tf.train.Saver(var_list=[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fnet"), global_step])
fnet_saver.restore(sess, tf.train.get_checkpoint_state(fnet_log_dir).model_checkpoint_path)
enet_saver = tf.train.Saver()
# enet_saver.restore(sess, tf.train.get_checkpoint_state(enet_log_dir).model_checkpoint_path)

# ============train=================

def trainFnet():
    if record_log:
        log_writer = tf.summary.FileWriter(fnet_log_dir, sess.graph)
    start_time = time.time()    
    
    while True:
        
        [ _ ] = sess.run([ fnet_train], feed_dict={ is_training: True})
    
        if global_step.eval() % 100 == 0 :
            print('step = %d, lr = %g, time = %g min' % (global_step.eval(), learning_rate.eval(), (time.time() - start_time) / 60.0))
            [  fnloss, summ] = sess.run([  fnet_loss , fnet_merged], feed_dict={is_training: False})
            print(fnloss)
            if record_log:
                log_writer.add_summary(summ, global_step.eval())
            print("==================")
            
        if global_step.eval() % 500 == 0 :
            if record_log: 
                fnet_saver.save(sess, os.path.join(fnet_log_dir, 'model.ckpt'), global_step.eval())

    print('total time = ', time.time() - start_time, 's')    
    if record_log:
        log_writer.close()    

    
def trainEnet():
    if record_log:
        log_writer = tf.summary.FileWriter(enet_log_dir, sess.graph)
    start_time = time.time()    
    
    while True:
        
        [ _ ] = sess.run([ enet_train], feed_dict={ is_training: True})
    
        if global_step.eval() % 100 == 0 :
            print('step = %d, lr = %g, time = %g min' % (global_step.eval(), learning_rate.eval(), (time.time() - start_time) / 60.0))
            [ acc , smloss, ctloss, ct_distance, summ] = sess.run([ accuarcy, softmax_loss, center_loss , center_distance, enet_merged], feed_dict={is_training: False})
            print(acc)
            print(smloss)
            print(ctloss)
            print(ct_distance)
            if record_log:
                log_writer.add_summary(summ, global_step.eval())
            print("==================")
            
        if global_step.eval() % 500 == 0 :
            if record_log: 
                enet_saver.save(sess, os.path.join(enet_log_dir, 'model.ckpt'), global_step.eval())

    print('total time = ', time.time() - start_time, 's')    
    if record_log:
        log_writer.close()    

# trainFnet()
trainEnet()
