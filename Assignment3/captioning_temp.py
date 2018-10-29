
# Write your own image captiong code
# You can modify the class structure
# and add additional function needed for image captionging

import tensorflow as tf
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Captioning():

    def __init__(self, is_training):
        self._start = 1
        
        self.is_training = is_training
        
        # for TRAINING
        self.epoch = 100
        self.batch_size = 2000
        
        self.n_hidden = 512
        
        self.n_words = 1004
        self.maxlen = 17
        self.dim_features = 512
        self.embedding_size = 512
        
        self.initializer = tf.random_uniform_initializer(minval= -0.08, maxval=  0.08)
      
    def batch_setting(self, batch_size):
        self.batch_size = batch_size
        
    def build_model(self):
         with tf.Graph().as_default() as self.graph:
            #loss = 0
            self.global_step_tensor = tf.contrib.framework.get_or_create_global_step()
 
            if self.is_training:
                self.img_features = tf.placeholder(tf.float32, shape=[self.batch_size, self.dim_features]) # img_features
                self.input_seqs = tf.placeholder(tf.int32, shape=[self.batch_size, self.maxlen - 1]) # captions[:, :16]
                self.target_seqs = tf.placeholder(tf.int32, shape=[self.batch_size, self.maxlen - 1]) # captions[:, 1:]
                self.input_mask = tf.placeholder(tf.int32, shape=[self.batch_size, self.maxlen - 1]) # mask = 1*(captions[:, 16]>0)
            else:
                self.img_features = tf.placeholder(tf.float32, shape=[1, self.dim_features])
                self.input_seqs = tf.placeholder(tf.int32, shape=[1, 1])
                self.batch_size = 1
            
            
            # IMAGE EMBEDDING
            '''
            with tf.variable_scope("image_embedding") as scope:
                self.image_embedding = tf.contrib.layers.fully_connected(
                                            inputs=self.img_features,
                                            num_outputs = self.embedding_size,
                                            activation_fn = None,
                                            weights_initializer = self.initializer,
                                            biases_initializer = None,
                                            scope = scope)
            '''                              
            self.image_embeddings = self.img_features #self.image_embedding # NEW IMAGE EMBEDDINGS
            # (500, 512)
            
            # SEQ EMBEDDING
            with tf.variable_scope('seq_embedding'):#, tf.device('/cpu:0'):
                # 'get_variable' can get existing values.
                self.embedding_map = tf.get_variable(name = 'map',
                                                shape = [self.n_words, self.embedding_size],
                                                initializer = self.initializer)
            self.seq_embeddings = tf.nn.embedding_lookup(self.embedding_map, self.input_seqs) # NEW SEQ EMBEDDINGS
            # (500, 16, 512)
            
            
            # LSTM MODEL
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.n_hidden, state_is_tuple = True);
            
            if self.is_training is True:
                self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell, input_keep_prob=0.7, output_keep_prob=0.7)
                
            with tf.variable_scope('lstm', initializer=self.initializer) as lstm_scope:
                self.zero_state = self.lstm_cell.zero_state(batch_size = self.batch_size, dtype = tf.float32)
                # PREDICT PHASE: self. INITIAL STATE
                _, self.initial_state = self.lstm_cell(self.image_embeddings, self.zero_state) # not using output value.
                lstm_scope.reuse_variables()
                
                
                if self.is_training is True:
                    seq_len = tf.reduce_sum(self.input_mask, 1)
                    lstm_output, state = tf.nn.dynamic_rnn(cell = self.lstm_cell,
                                                       inputs = self.seq_embeddings,
                                                       sequence_length = seq_len,
                                                       initial_state = self.initial_state,
                                                       dtype = tf.float32,
                                                       scope = lstm_scope)
                else:
                    # (512 + 512) = (c + h)
                    self.state_feed = tf.placeholder(dtype=tf.float32, shape=[None, sum(self.lstm_cell.state_size)], name="state_feed")
                    # (512, 512)
                    state_tuple = tf.split(value=self.state_feed, num_or_size_splits=2, axis=1)

                    # Run a single LSTM step.
                    # PREDICT PHASE: self. STATE_TUPLE
                    lstm_output, self.state_tuple = self.lstm_cell(inputs=tf.squeeze(self.seq_embeddings, axis=1), state=state_tuple)
                    #lstm_output, state_tuple = lstm_cell(inputs=self.seq_embeddings[:,0,:], state=initial_state)

            self.lstm_output = tf.reshape(lstm_output, [-1, self.lstm_cell.output_size]) # (500, 512)
        
            
            # for LOSS with MASK
            if self.is_training is True:
                weights = tf.to_float(tf.reshape(self.input_mask,[-1])) # MASK WEIGHTS
                targets = tf.reshape(self.target_seqs, [ -1]) # TARGET
        
            with tf.variable_scope('logits') as logits_scope:
                self.logits = tf.contrib.layers.fully_connected(
                    inputs = self.lstm_output, 
                    num_outputs = self.n_words,
                    activation_fn = None, 
                    weights_initializer = self.initializer, 
                    scope = logits_scope)
                
            if self.is_training is True:
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets, logits = self.logits)
                self.batch_loss = tf.div(tf.reduce_sum(tf.multiply(self.loss, weights)), tf.reduce_sum(weights), name = 'batch_loss')
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=(1e-3), epsilon=1e-3)
            
                self.train_op = self.optimizer.minimize(self.batch_loss, global_step=self.global_step_tensor)
                
            else:
                self.soft_logits = tf.nn.softmax(self.logits, name="softmax")
                self.predict = tf.argmax(self.soft_logits, axis=1)
                
                
             
    def train(self, img_features, captions):
        tf.reset_default_graph()
        
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)), graph=self.graph) as sess:
            
            init = tf.global_variables_initializer()
            sess.run(init)
            
            total_batch = int(img_features.shape[0] // self.batch_size)     
            
            print("EPOCH: %d" % self.epoch)
            for e in range(self.epoch):
                print("LOCAL EPOCH: %d" % e)
                for b in range(total_batch):
                    before = b * self.batch_size
                    after = before + self.batch_size
                    _, loss, a = sess.run([self.train_op, self.batch_loss, self.lstm_output],
                                              feed_dict={
                                                 self.img_features: img_features[before: after],
                                                 self.input_seqs: captions[before: after, :16],
                                                 self.target_seqs: captions[before: after, 1:],
                                                 self.input_mask: 1 * (captions[before: after, :16] > 0)
                                                 })
                if e % 10 == 0:
                    print("loss: %.4f" % (loss))
                # SAVE the MODEL
                #saver = tf.train.Saver(max_to_keep=3)
                #saver.save(sess, "./models/my_model_f", global_step=self.global_step_tensor.eval())
                
            print("Training done!")

            
    def prediction(self, sess, img_feature, word_to_idx, idx_to_word): # img_feature: (512,)
        
        captions = []
        end_mark = word_to_idx["<END>"]
        
        
        for e in range(img_feature.shape[0]):
            caption = []
            
            initial_state = sess.run(self.initial_state, feed_dict={self.img_features: img_feature[e: e+1]})
            state = np.hstack((initial_state.c, initial_state.h))
        
            word = word_to_idx["<START>"]
        
            for _ in range(self.maxlen - 1):
                caption.append(word)
                if word is end_mark:
                    break

                word, state = sess.run([self.predict, self.state_tuple],
                                       feed_dict={
                                           self.state_feed: state,
                                           self.input_seqs: np.array([[word]])})
                state = np.hstack((state.c, state.h))
                word = word[0]
                
            captions.append(caption)
            
        return np.array(captions)
        
        
        
