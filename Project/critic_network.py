
# coding: utf-8

from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

import tensorflow as tf 
import numpy as np
import math


LAYER1_SIZE = 300
LAYER2_SIZE = 600
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.00001
L2 = 0.0

class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,state_dim,action_dim):
        self.time_step = 0
        self.sess = sess
        # create q network
        self.state_input,self.action_input,self.q_value_output,self.net,self.is_training = self.create_q_network(state_dim,action_dim)

        # create target q network (the same structure with q network)
        self.target_state_input,        self.target_action_input,        self.target_q_value_output,        self.target_update,        self.target_is_training = self.create_target_q_network(state_dim,action_dim,self.net)

        self.create_training_method()

        # initialization 
        self.sess.run(tf.global_variables_initializer())

        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float",[None,1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) #+ weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.parameters_gradients = self.optimizer.compute_gradients(self.cost)

        for i, (grad,var) in enumerate(self.parameters_gradients):
            if grad is not None:
                self.parameters_gradients[i] = (tf.clip_by_norm(grad, 5.0),var)

        self.train_op = self.optimizer.apply_gradients(self.parameters_gradients)
        self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

    def create_q_network(self,state_dim,action_dim):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float",[None,state_dim])
        action_input = tf.placeholder("float",[None,action_dim])
        is_training = tf.placeholder(tf.bool)

        W1 = self.variable([state_dim,layer1_size],state_dim)
        b1 = self.variable([layer1_size],state_dim)
        W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
        W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
        b2 = self.variable([layer2_size],layer1_size+action_dim)
        W3 = tf.Variable(tf.random_uniform([layer2_size,1],-1e-4,1e-4))
        b3 = tf.Variable(tf.random_uniform([1],-1e-4,1e-4))

        layer0_bn = self.batch_norm_layer(state_input,training_phase=is_training,scope_bn='q_batch_norm_0',activation=tf.identity)

        layer1 = tf.nn.relu(tf.matmul(layer0_bn,W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
        q_value_output = tf.identity(tf.matmul(layer2,W3) + b3)

        return state_input,action_input,q_value_output,[W1,b1,W2,W2_action,b2,W3,b3],is_training

    def create_target_q_network(self,state_dim,action_dim,net):
        state_input = tf.placeholder("float",[None,state_dim])
        action_input = tf.placeholder("float",[None,action_dim])
        is_training = tf.placeholder(tf.bool)

        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]
        layer0_bn = self.batch_norm_layer(state_input,training_phase=is_training,scope_bn='target_q_batch_norm_0',activation=tf.identity)

        layer1 = tf.nn.relu(tf.matmul(layer0_bn,target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
        q_value_output = tf.identity(tf.matmul(layer2,target_net[5]) + target_net[6])

        return state_input,action_input,q_value_output,target_update,is_training

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,y_batch,state_batch,action_batch):
        self.time_step += 1
        self.sess.run(self.train_op,feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch,
            self.action_input:action_batch,
            self.is_training: True
            })

    def gradients(self,state_batch,action_batch):
        return self.sess.run(self.action_gradients,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch,
            self.is_training: False
            })[0]

    def target_q(self,state_batch,action_batch):
        return self.sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch,
            self.target_is_training: False
            })

    def q_value(self,state_batch,action_batch):
        return self.sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch,
            self.is_training: False})

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

    def batch_norm_layer(self,x,training_phase,scope_bn,activation=None):
        return tf.cond(training_phase, 
        lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
        updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
        lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
        updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))

