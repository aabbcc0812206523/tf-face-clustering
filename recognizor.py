
import tensorflow as tf
import numpy
import math
from tfobjs import *

class ClusterNet:
    def __init__(self, input_tensor, feature_dimensions, class_nums, is_training):
  
        with tf.variable_scope('layer1'):
            self.layer1 = FcObj()
            self.layer1.set_input(input_tensor)
            self.layer1.fc(feature_dimensions, l2=0.0)
            self.layer1.set_output(parametric_relu(self.layer1.logit))
            
        with tf.variable_scope('layer2'):
            self.layer2 = FcObj()
            self.layer2.set_input(self.layer1.output)
            self.layer2.fc(class_nums, l2=0.0)
            self.layer2.set_output(self.layer2.logit)
            
            
class EncodeNet:

    def __init__(self, input_tensor, is_training):
        
       
        with tf.variable_scope('in_layer1'):
            self.ilayer1 = ConvObj()
            self.ilayer1.set_input(input_tensor)
            self.ilayer1.batch_norm(self.ilayer1.conv2d([5, 5], 64, [1, 2, 2, 1]), is_training=is_training)
            self.ilayer1.set_output(parametric_relu(self.ilayer1.bn))
        # ===================64->32

        with tf.variable_scope('in_layer2'):
            self.ilayer2 = ConvObj()
            self.ilayer2.set_input(self.ilayer1.output)
            self.ilayer2.batch_norm(self.ilayer2.conv2d([5, 5], 128, [1, 2, 2, 1]), is_training=is_training)
            self.ilayer2.set_output(parametric_relu(self.ilayer2.bn))
        # ===================32->16

        with tf.variable_scope('in_layer3'):
            self.ilayer3 = ConvObj()
            self.ilayer3.set_input(self.ilayer2.output)
            self.ilayer3.batch_norm(self.ilayer3.conv2d([5, 5], 256, [1, 2, 2, 1]), is_training=is_training)
            self.ilayer3.set_output(parametric_relu(self.ilayer3.bn))
        # ===================16->8

        with tf.variable_scope('in_layer4'):
            self.ilayer4 = ConvObj()
            self.ilayer4.set_input(self.ilayer3.output)
            self.ilayer4.batch_norm(self.ilayer4.conv2d([5, 5], 512, [1, 2, 2, 1]), is_training=is_training)
            self.ilayer4.set_output(parametric_relu(self.ilayer4.bn))
        # ===================8->4
        
        with tf.variable_scope('in_layer5'):
            self.ilayer5 = ConvObj()
            self.ilayer5.set_input(self.ilayer4.output)
            self.ilayer5.batch_norm(self.ilayer5.conv2d([4, 4], 1024, [1, 1, 1, 1], padding='VALID'), is_training=is_training)
            self.ilayer5.set_output(parametric_relu(self.ilayer5.bn))
        # ===================4->1
        
        with tf.variable_scope('out_layer0'):
            self.olayer0 = FcObj()
            self.olayer0.set_input(self.ilayer5.output)
            self.olayer0.batch_norm(self.olayer0.fc(4 * 4 * 512, l2=0.0), is_training=is_training)
            self.olayer0.set_output(parametric_relu(self.olayer0.bn))
        # ===================1->4
        
        with tf.variable_scope('out_layer1'):
            self.olayer1 = ConvObj()
            self.olayer1.set_input(tf.reshape(self.olayer0.output, [-1, 4, 4, 512]))
            self.olayer1.batch_norm(self.olayer1.deconv2d([5, 5], [8, 8, 256], strides=[1, 2, 2, 1]), is_training=is_training)
            self.olayer1.set_output(parametric_relu(self.olayer1.bn))
        # ===================4->8

        with tf.variable_scope('out_layer2'):
            self.olayer2 = ConvObj()
            self.olayer2.set_input(self.olayer1.output)
            self.olayer2.batch_norm(self.olayer2.deconv2d([5, 5], [16, 16, 128], strides=[1, 2, 2, 1]), is_training=is_training)
            self.olayer2.set_output(parametric_relu(self.olayer2.bn))
        # ===================8->16
    
        with tf.variable_scope('out_layer3'):
            self.olayer3 = ConvObj()
            self.olayer3.set_input(self.olayer2.output)
            self.olayer3.batch_norm(self.olayer3.deconv2d([5, 5], [32, 32, 64], strides=[1, 2, 2, 1]), is_training=is_training)
            self.olayer3.set_output(parametric_relu(self.olayer3.bn))
        # ===================16->32
    
        with tf.variable_scope('out_layer4'):
            self.olayer4 = ConvObj()
            self.olayer4.set_input(self.olayer3.output)
            self.olayer4.batch_norm(self.olayer4.deconv2d([5, 5], [64, 64, 3], strides=[1, 2, 2, 1]), is_training=is_training)
            self.olayer4.set_output(tf.nn.tanh(self.olayer4.logit))
        # ===================32->64
        
        self.feature_output = self.ilayer5.output
