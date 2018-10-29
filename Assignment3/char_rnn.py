# This library is used in Assignment3_Part3_CharRNN
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
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
        # choose different rnn cell 
        if args.model == 'rnn':
            cell_fn = rnn.RNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        # These are some primers that you must implement: explicitly used at the training loop
        # the nomenclature is from the original code for less headaches.
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
        
        # softmax output layer, use softmax to classify
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        # transform input to embedding
        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        # unstack the input to fits in rnn model
        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        self.final_state = last_state

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.targets, [-1])],[tf.ones([args.batch_size * args.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        
                                             



        # the original source code initializes self.lr = tf.Variable(0.0, trainable=False)
        # why zero?! does the model not learn at all? This is for a technical factoring of the original source code
        # the code later assigns the lr in the training loop by sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
        # so let's just use this scheme for less confusions
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # calculate gradients
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # And these member variables are explicitly used from sample() function call below from the original code
        # hint: feeling lost? try googling what member functions self.cell is calling, like zero_state for example.
        # we always do it right?
        # and ofc, if you feel that these variables are ugly, you can just use your code scheme instead of these ones.
        # just make sure that the sample() works
        self.probs = tf.nn.softmax(self.logits)

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
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for _ in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]
            sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret        
    