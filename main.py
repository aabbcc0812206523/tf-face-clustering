import time
import tensorflow as tf
import numpy
import os
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import convert_to_tensor


import data_muctface
import data_celebaface

from recognizor import ClusterNet, EncodeNet

from tensorflow.contrib.keras.python.keras.activations import softmax
from tensorflow.contrib.learn.python.learn.trainable import Trainable

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True


record_log = True
enet_log_dir = './log/enet_celeba/'
cnet_log_dir = './log/cnet_muct_128/'
embedding_log_dir = './log/embedding_128/'

is_training = tf.placeholder(tf.bool, name='ph_is_training')
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(0.001, global_step, 500, 1, staircase=False, name='learning_rate')


# ========================
batch_size = 64

[image64_muct, label64_muct] = data_muctface.inputs(False, './muct-faces/', batch_size)  # -1.~1.
image64_muct = image64_muct / 127.5 - 1.0
label64_muct = tf.reshape(tf.cast(label64_muct, tf.int32), [-1])

image64_celeba = data_celebaface.inputs('./celeba-faces/', batch_size)  # -1.~1.
image64_celeba = image64_celeba / 127.5 - 1.0




with tf.variable_scope('enet'):
    enet_celeba = EncodeNet(image64_celeba, is_training=is_training)
with tf.variable_scope('enet', reuse=True):    
    enet_muct = EncodeNet(image64_muct, is_training=is_training)

with tf.variable_scope('cnet'):
    cnet = ClusterNet(enet_muct.feature_output, 128, data_muctface.CLASS_NUMS, is_training=is_training)


with tf.variable_scope('loss'):
    enet_loss = tf.reduce_mean(tf.square(enet_celeba.olayer4.output - image64_celeba)) * 127.5
    
    weight_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label64_muct, logits=cnet.layer2.logit))
    accuarcy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(cnet.layer2.logit, tf.reshape(label64_muct, [-1]) , 1), tf.float32))
    
    centers = tf.get_variable('centers', [cnet.layer2.output_size, cnet.layer1.output_size], dtype=tf.float32,
            initializer=tf.zeros_initializer(tf.float32), trainable=False)
    center_distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(centers), axis=1)))
    center_loss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(cnet.layer1.output - tf.gather(centers, label64_muct)), axis=1)))

# ===================

enet_train = tf.train.AdamOptimizer(learning_rate).minimize(
    enet_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="enet") , global_step=global_step)
cnet_train = tf.train.AdamOptimizer(learning_rate).minimize(
        softmax_loss + center_loss + weight_loss, var_list=[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnet"), centers ], global_step=global_step)

# ===================

enet_merged = tf.summary.merge([tf.summary.scalar('enet_loss', enet_loss), tf.summary.image('enet_restore_image', (enet_celeba.olayer4.output + 1) / 2, 64)])
cnet_merged = tf.summary.merge([tf.summary.scalar('softmax_loss', softmax_loss), tf.summary.scalar('center_loss', center_loss),
                                tf.summary.scalar('center_distance', center_distance)])

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()

# ========restore===============
enet_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="enet") + [global_step])
cnet_saver = tf.train.Saver()

# ============train=================

def trainEnet():
    if record_log:
        log_writer = tf.summary.FileWriter(enet_log_dir, sess.graph)
    start_time = time.time()    
    
    while True:
        
        [ _ ] = sess.run([ enet_train], feed_dict={ is_training: True})
    
        if global_step.eval() % 100 == 0 :
            print('step = %d, lr = %g, time = %g min' % (global_step.eval(), learning_rate.eval(), (time.time() - start_time) / 60.0))
            [  fnloss, summ] = sess.run([  enet_loss , enet_merged], feed_dict={is_training: False})
            print(fnloss)
            if record_log:
                log_writer.add_summary(summ, global_step.eval())
            print("==================")
            
        if global_step.eval() % 500 == 0 :
            if record_log: 
                enet_saver.save(sess, os.path.join(enet_log_dir, 'model.ckpt'), global_step.eval())

    print('total time = ', time.time() - start_time, 's')    
    if record_log:
        log_writer.close()    

    
def trainCnet():
    if record_log:
        log_writer = tf.summary.FileWriter(cnet_log_dir, sess.graph)
    start_time = time.time()    
    
    while True:
        
        [ _ ] = sess.run([ cnet_train], feed_dict={ is_training: True})
    
        if global_step.eval() % 100 == 0 :
            print('step = %d, lr = %g, time = %g min' % (global_step.eval(), learning_rate.eval(), (time.time() - start_time) / 60.0))
            [ acc , smloss, ctloss, ct_distance, summ] = sess.run([ accuarcy, softmax_loss, center_loss , center_distance, cnet_merged], feed_dict={is_training: False})
            print(acc)
            print(smloss)
            print(ctloss)
            print(ct_distance)
            if record_log:
                log_writer.add_summary(summ, global_step.eval())
            print("==================")
            
        if global_step.eval() % 500 == 0 :
            if record_log: 
                cnet_saver.save(sess, os.path.join(cnet_log_dir, 'model.ckpt'), global_step.eval())

    print('total time = ', time.time() - start_time, 's')    
    if record_log:
        log_writer.close()    

def createEmbedding(embedding_batch_times = 10):
    embedding_var = numpy.zeros([embedding_batch_times * batch_size, cnet.layer1.output_size])
    log_writer = tf.summary.FileWriter(embedding_log_dir, sess.graph) 
    tsvfile = open(os.path.join(embedding_log_dir, 'labels.tsv') , mode='w')
    for i in range(embedding_batch_times):
        [data, lbl] = sess.run([ cnet.layer1.output, label64_muct], feed_dict={ is_training: False})
        for num in lbl:
            tsvfile.write('%d\n' % num)
        embedding_var[i * batch_size:(i + 1) * batch_size] = data

    tsvfile.close();
    embedding_var = tf.Variable(embedding_var, name='embedding_var')
    sess.run(tf.global_variables_initializer())
    embedding_saver = tf.train.Saver(var_list=[embedding_var])
    embedding_saver.save(sess, os.path.join(embedding_log_dir, 'model.ckpt'))
    log_writer.close()
    print(embedding_var.shape.as_list())

#trainEnet()

#enet_saver.restore(sess, tf.train.get_checkpoint_state(enet_log_dir).model_checkpoint_path)
#sess.run(global_step.assign(0))
#trainCnet()

cnet_saver.restore(sess, tf.train.get_checkpoint_state(cnet_log_dir).model_checkpoint_path)
createEmbedding(50)
