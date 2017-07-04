import time
import tensorflow as tf
import numpy
import os



import muctface_input

from recognizor import EncodeNet, FeatureNet


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True



enet_log_dir = './log/enet_muct_128/'
fnet_log_dir = './log/fnet_celeba/'
embedding_log_dir = './log/embedding/'


is_training = tf.placeholder(tf.bool, name='ph_is_training')
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(0.001, global_step, 500, 1, staircase=False, name='learning_rate')


# ========================
batch_size = 640
[image64_muct, label64_muct] = muctface_input.inputs(False, './muct-faces/', batch_size)  # -1.~1.
image64_muct = image64_muct / 127.5 - 1.0
label64_muct = tf.reshape(tf.cast(label64_muct, tf.int32), [-1])


with tf.variable_scope('fnet'):
    fnet = FeatureNet(image64_muct, is_training=is_training)

with tf.variable_scope('enet'):
    enet = EncodeNet(fnet.feature_output, 128, muctface_input.CLASS_NUMS, is_training=is_training)

centers = tf.get_variable('loss/centers', [enet.layer2.output_size, enet.layer1.output_size], dtype=tf.float32,
            initializer=tf.zeros_initializer(tf.float32), trainable=False)
center_distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(centers), axis=1)))
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()



# ========restore===============
fnet_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fnet"))
# fnet_saver.restore(sess, tf.train.get_checkpoint_state(fnet_log_dir).model_checkpoint_path)
enet_saver = tf.train.Saver()
enet_saver.restore(sess, tf.train.get_checkpoint_state(enet_log_dir).model_checkpoint_path)
temp = tf.sqrt(tf.reduce_sum(tf.square(centers.eval()[0:100] - centers.eval()[100:200]), axis=1)).eval()
# print(temp)
print(numpy.mean(temp))
print(numpy.std(temp))
print(temp.shape)
centers_batch = tf.gather(centers, label64_muct)
center_loss = tf.sqrt(tf.reduce_sum(tf.square(enet.layer1.output - centers_batch), axis=1))
[temp] = sess.run([ center_loss], feed_dict={ is_training: False})
# print(temp)
print(numpy.mean(temp))
print(numpy.std(temp))
print(temp.shape)


temp = enet.layer1.output * tf.reshape(tf.cast(tf.equal(label64_muct, 39), tf.float32), [-1, 1])
#temp = enet.layer1.output
[lbl , temp] = sess.run([label64_muct, temp], feed_dict={ is_training: False})
preIndex = -1;
for i in range(temp.shape[0]):
    if (numpy.std(temp[i]) == 0):
        continue
    if (preIndex == -1):
        preIndex = i
        continue
    print('%g\t\t%d--%d' % (numpy.sqrt(numpy.sum(numpy.square(temp[i] - temp[preIndex]))), i, preIndex))
    preIndex = i
    

# ===================================
def createEmbedding():
    embedding_batch_times = 5
    embedding_var = numpy.zeros([embedding_batch_times * batch_size, enet.layer1.output_size])
    log_writer = tf.summary.FileWriter(embedding_log_dir, sess.graph) 
    tsvfile = open(os.path.join(embedding_log_dir, 'labels.tsv') , mode='w')
    for i in range(embedding_batch_times):
        [data, lbl] = sess.run([ enet.layer1.output, label64_muct], feed_dict={ is_training: False})
        for num in lbl:
            tsvfile.write('%d\n' % num)
        embedding_var[i * batch_size:(i + 1) * batch_size] = data

    tsvfile.close();
    embedding_var = tf.Variable(embedding_var, name='embedding_var')
    sess.run(tf.global_variables_initializer())
    embedding_saver = tf.train.Saver(var_list=[embedding_var])
    embedding_saver.save(sess, os.path.join(embedding_log_dir, 'model.ckpt'))
    print(embedding_var.shape.as_list())
    
    log_writer.close()
createEmbedding()

