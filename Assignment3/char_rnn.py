# This library is used in Assignment3_Part3_CharRNN
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# RNN model class definition
# the class contains all the TF ops that define the recurrent neural networks computation
class Model():
    def __init__(self, args, training=True):
        # get some args
        self.args = args
        # after the end of training, the model will just use batch_size=1 and seq_length=1 for easier code
        # and you don't need to do anything because sample_eval() for evaluation is already given
        if not training:
            args.batch_size = 1
            args.seq_length = 1
            
        """
        Implement your CharRNN from here: Construct all the TF ops for CharRNN inside this initializer
        hint: looking at arguments passed to this __init__ is a good place to start
        remember: try to implement the model yourself first, and consider the original code as a last resort
        we all copy someone's code sure, but think hard before you copy. it's good for you. i mean REALLY.
        """
        
        # These are some primers that you must implement: explicitly used at the training loop
        # the nomenclature is from the original code for less headaches.
        self.initial_state = None
        self.input_data = None
        self.targets = None
        self.cost = None
        self.final_state = None
        self.train_op = None

        # the original source code initializes self.lr = tf.Variable(0.0, trainable=False)
        # why zero?! does the model not learn at all? This is for a technical factoring of the original source code
        # the code later assigns the lr in the training loop by sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
        # so let's just use this scheme for less confusions
        self.lr = tf.Variable(0.0, trainable=False)

        # And these member variables are explicitly used from sample() function call below from the original code
        # hint: feeling lost? try googling what member functions self.cell is calling, like zero_state for example.
        # we always do it right?
        # and ofc, if you feel that these variables are ugly, you can just use your code scheme instead of these ones.
        # just make sure that the sample() works
        self.probs = None
        self.cell = None

    def sample(self, sess, chars, vocab, num=200, prime='The '):
        """
        implement your sampling method from here: give a model an ability to spit out characters from RNN's hidden state.
        You are given the following parameters:
        1. self & sess (obviously)
        2. vocabulary data (chars & vocab)
        3. num: the number of characters that the model will spit out. This will given by args.n inside the evaluation code.
        4. prime: primer string that will be fed to the model before the actual sampling.
                  the default from evaluation code is blank (u'') for freedom! (without human control)
                  
        there are various ways to sample from the hidden state:
        you can be greedy: just pick the character that has the highest probability for each step.
        you can be stochastic: sample the character from the probability distribution of characters for each step.
        or, just be creative and implement your own unique sampling method. your call!
        """
        
        # the final sentence is defined "ret" in the original source code.
        ret = None
        return ret
    