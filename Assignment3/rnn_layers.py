# This library is used for Assignment3_Part1_Implementing_RNN

import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""

#------------------------------------------------------
# vanilla rnn step forward
#------------------------------------------------------
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##########################################################################
    a = prev_h.dot(Wh) + x.dot(Wx) + b
    next_h = np.tanh(a)
    cache = (x,prev_h,Wh,Wx,b,next_h)
    ##########################################################################
    #                               END OF YOUR CODE                             #
    ##########################################################################
    return next_h, cache


#------------------------------------------------------
# vanilla rnn step backward 
#------------------------------------------------------
def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##########################################################################
    x, prev_h,Wh,Wx,b,next_h = cache
    da = dnext_h * (1 - next_h * next_h)
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    db = np.sum(da,axis = 0)

    ##########################################################################
    #                               END OF YOUR CODE                             #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


#------------------------------------------------------
# vanilla rnn forward
#------------------------------------------------------
def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##########################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above.                                                                     #
    ##########################################################################
    N,T,D = x.shape
    H = b.shape[0]
    h = np.zeros((N,T,H))
    prev_h = h0
    for i in range(T) :
        xi = x[:,i,:]
        next_h,_ = rnn_step_forward(xi,prev_h,Wx,Wh,b)
        prev_h = next_h
        h[:,i,:] = prev_h
    cache = (x,h0,Wh,Wx,b,h)

    ##########################################################################
    #                               END OF YOUR CODE                             #
    ##########################################################################
    return h, cache


#------------------------------------------------------
# vanilla rnn backward 
#------------------------------------------------------
def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    ##########################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above.                                                             #
    ##########################################################################
    x,h0,Wh,Wx,b,h = cache
    N,T,H = dh.shape
    D = x.shape[2]
    
    next_h = h[:,T-1,:]
    
    #initialize
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx= np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))

    for i in range(T) :
        t = T-1-i
        xi = x[:,t,:]
        
        if t == 0 :
            prev_h = h0
        else :
            prev_h = h[:,t-1,:]
        
        cache_layer = (xi,prev_h,Wh,Wx,b,next_h)
        next_h = prev_h
        dnext_h = dh[:,t,:] + dprev_h
        dx[:,t,:], dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_h, cache_layer)
        dWx, dWh, db = dWx+dWxt, dWh+dWht, db+dbt
    
    dh0 = dprev_h

    ##########################################################################
    #                               END OF YOUR CODE                             #
    ##########################################################################
    return dx, dh0, dWx, dWh, db



def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def getdata_rnn_step_forward():
	np.random.seed(2177)

	N, D, H = 3, 5, 4
	x = np.random.randn(N, D)
	prev_h = np.random.randn(N, H)
	Wx = np.random.randn(D, H)
	Wh = np.random.randn(H, H)
	b = np.random.randn(H)

	expt_next_h = np.asarray([
		[-0.99921173, -0.99967951,  0.39127099, -0.93436299],
	 	[ 0.84348286,  0.99996526, -0.9978802,   0.99996645],
	  [-0.94481752, -0.71940178,  0.99994009, -0.64806562]])

	return x, prev_h, Wx, Wh, b, expt_next_h


def getdata_rnn_step_backward(x,h,Wx,Wh,b,dnext_h):

	fx  = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
	fh  = lambda prev_h: rnn_step_forward(x, h, Wx, Wh, b)[0]
	fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
	fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
	fb  = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]

	dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
	dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
	dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
	dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
	db_num = eval_numerical_gradient_array(fb, b, dnext_h)

	return dx_num, dprev_h_num, dWx_num, dWh_num, db_num


def getdata_rnn_forward():
	np.random.seed(2177)

	N, D, T, H = 2,3,4,5
	x = np.random.randn(N, T, D)
	h0 = np.random.randn(N, H)
	Wx = np.random.randn(D, H)
	Wh = np.random.randn(H, H)
	b = np.random.randn(H)

	expt_next_h = np.asarray([
	[[ 0.79899136, -0.90076473, -0.69325878, -0.99991011,  0.92991908],
   [-0.04474799, -0.99999994, -0.72167573, -0.99942462, -0.98397185],
   [ 0.98674954, -0.74668554, -0.30836793, -0.87580427, -0.25076433],
   [ 0.99999994,  0.46495278, -0.6291276 ,  0.44811995, -0.91013617]],

 	[[-0.57789921, -0.10875688, -0.99049558, -0.58448393,  0.76942269],
   [-0.05646372, -0.99855467, -0.827688  , -0.65262183, -0.98211725],
   [ 0.89687939,  0.99998112, -0.99999517,  0.66932722,  0.99952606],
   [-0.97608409, -0.64972242, -0.99987169, -0.99747724,  0.99962792]]])

	return x, h0, Wx, Wh, b, expt_next_h

def getdata_rnn_backward(x,h0,Wx,Wh,b,dout):
	fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
	fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
	fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
	fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
	fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]

	dx_num = eval_numerical_gradient_array(fx, x, dout)
	dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
	dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
	dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
	db_num = eval_numerical_gradient_array(fb, b, dout)
	
	return dx_num, dh0_num, dWx_num, dWh_num, db_num


