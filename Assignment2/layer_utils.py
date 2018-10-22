import numpy as np
from Utils.data_utils import plot_conv_images

def conv_forward(x, w, b, conv_param):
    """
    Computes the forward pass for a convolutional layer.

    Inputs:
    - x: Input data, of shape (N, H, W, C) 
        N : Input number
        H : Height
        W : Width
        C : Channel
    - w: Weights, of shape (F, WH, WW, C)
        F : Filter number
        WH : Filter height
        WW : Filter width
        C : Channel
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields, of shape (1, SH, SW, 1)
          SH : Stride height
          SW : Stride width
      - 'padding': "valid" or "same". "valid" means no padding.
        "same" means zero-padding the input so that the output has the shape as (N, ceil(H / SH), ceil(W / SW), F)
        If the padding on both sides (top vs bottom, left vs right) are off by one, the bottom and right get the additional padding.
         
    Outputs:
    - out: Output data
    - cache: (x, w, b, conv_param)
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    pass
    #First, initialize variables of input data(x), weights, bias, conv_param(stride and padding).
    #Initializing x from the description of x above (Input number, Height, Width, Channel),
    N, H, W, C = x.shape
    
    #Next, initialize w from the description of w above (Filter number, Filter height, Filter width, Channel),
    F, WH, WW, C = w.shape
    
    #Next, initialze stride and padding in conv_param. As it says it is a dictionary with keys.
    #So we need to take keys(stride and padding) from dictionary and initialize variables of strides(Stride height, Stride width) and padding(same, valid) as below,
    _, SH, SW, _= conv_param['stride']
    #In cov_param dictionary, padding key will have either same or valid.
    padding = conv_param['padding']
    
    #We need to define if statement for padding, because it will have 'same' or 'valid' value in padding key.
    #So if we have 'same' value in padding,
    #We need to have shape of the output like (N, ceil(H / SH), ceil(W / SW), F)
    #which means height and width of the output will be same as on slide 55, 57 of CNN lecture note.
    if padding == 'same':
        #For ceiling, we can use double slash.
        #Referenced from https://stackoverflow.com/questions/32558805/ceil-and-floor-equivalent-in-python-3-without-math-module
        #output_height = (H + 2 - WH) // SH + 1
        #output_width = (W + 2 - WW) // SW + 1
        #With above equation, we get error that 'ValueError: operands could not be broadcast together with shapes (2,2,2,2) (2,2,3,2)'
        #in conv_forward(same). because we think width is different.
        #So we changed litte bit, also changed same in conv_backward function
        output_height = (H - 1) // SH + 1
        output_width = (W - 1) // SW + 1
        
        #As it says bottom and right side get additional padding,
        #Referenced from slide 55 of CNN lecture note and slide 10 assignment 2 note
        #For making x with padding, used the reference "https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python"
        #Also, referenced from numpy.pad library (https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.pad.html)
        x_pad_width = ((0, 0), ((WH - 1) // 2, WH // 2), ((WW - 1) // 2, WW // 2), (0, 0))
        x_with_pad = np.pad(x, x_pad_width, 'constant')
        
        #Initialize output as empty.
        out = np.empty((N, output_height, output_width, F))
        #Now, we need to do work for CNN, that means we have to make for loops.
        #1. Input number index(input_idx)
        for input_idx in range(N):
            #2. Filter Index(filter_idx)
            for filter_idx in range(F):
                #3. Initialize looping for height first because filter moves direction right to left. (i)
                for i in range(output_height):
                    #4. Next, initialize looping for width (j)
                    for j in range(output_width):
                        #Defines output with summation with input and weight (np.sum) and adding bias
                        out[input_idx, i, j, filter_idx] = np.sum(x_with_pad[input_idx, i * SH:i * SH + WH, j * SW:j * SW + WW, :] * w[filter_idx, :, :, :]) + b[filter_idx]

        #print ("out will be like (our code)", out)
    #Else if padding is valid, there is no padding so we need to make height and width of the output like
    #(m - k) / s + 1 which means in here, (H - WH) // SH + 1 and (W - WW) // SW + 1
    elif padding == 'valid':
        output_height = (H - WH) // SH + 1
        output_width = (W - WW) // SW + 1
        #Initialize output as empty.
        out = np.empty((N, output_height, output_width, F))
        #Now, we need to do work for CNN, that means we have to make for loops.
        #1. Input number index(input_idx)
        for input_idx in range(N):
            #2. Filter Index(filter_idx)
            for filter_idx in range(F):
                #3. Initialize looping for height first because filter moves direction right to left. (i)
                for i in range(output_height):
                    #4. Next, initialize looping for width (j)
                    for j in range(output_width):
                        #Defines output with np.sum and adding bias
                        out[input_idx, i, j, filter_idx] = np.sum(x[input_idx, i * SH:i * SH + WH, j * SW:j * SW + WW, :] * w[filter_idx, :, :, :]) + b[filter_idx]

        #print ("out will be like (our code)", out)
    
    
    #Done getting output(out)

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    cache = (x, w, b, conv_param)
    return out, cache
    

def conv_backward(dout, cache):
    """
    Computes the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Outputs:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    pass
    #First we need to get variables(x, w, b, conv_param) from cache which is returned from conv_forward function.
    x, w, b, conv_param = cache
    #And then, it is same as conv_forward function for setting variables and making height and width for output.
    
    #Initializing x from the description of x above (Input number, Height, Width, Channel),
    N, H, W, C = x.shape
    
    #Next, initialize w from the description of w above (Filter number, Filter height, Filter width, Channel),
    F, WH, WW, C = w.shape
    
    #Next, initialze stride and padding in conv_param. As it says it is a dictionary with keys.
    #So we need to take keys(stride and padding) from dictionary and initialize variables of strides(Stride height, Stride width) and padding(same, valid) as below,
    _, SH, SW, _= conv_param['stride']
    #In cov_param dictionary, padding key will have either same or valid.
    padding = conv_param['padding']
    
    #We need to define if statement for padding, because it will have 'same' or 'valid' value in padding key.
    #So if we have 'same' value in padding,
    #We need to have shape of the output like (N, ceil(H / SH), ceil(W / SW), F)
    #which means height and width will be changed as it says on slide 55, 57 of CNN lecture note.
    if padding == 'same':
        #For ceiling, we can use double slash.
        #Referenced from https://stackoverflow.com/questions/32558805/ceil-and-floor-equivalent-in-python-3-without-math-module
        output_height = (H - 1) // SH + 1
        output_width = (W - 1) // SW + 1
        #As it says bottom and right side get additional padding,
        #Referenced from slide 55 of CNN lecture note and slide 10 assignment 2 note
        #For making x with padding, used the reference "https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python"
        #Also, referenced from numpy.pad library (https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.pad.html)
        #In order with top, bottom, left, right
        x_pad_width = ((0, 0), ((WH - 1) // 2, WH // 2), ((WW - 1) // 2, WW // 2), (0, 0))
        x_with_pad = np.pad(x, x_pad_width, 'constant')
        
        #Now, we need to initialize each dx, dw, db variables with same shape as x, w, b
        #Because we need the gradient with respect to each data index.
        #We were going to use empty_like but it gives me error.
        #It is ok using in conv_forward, but can't be in conv_backward.
        #because the difference between zeros_like and empty is both empty_like and zeros_like create same shape,
        #but empty_like doesn't initialize the returned array (https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.empty_like.html)
        
        #First, dx is gradient with respect to x_with_pad
        dx = np.zeros_like(x_with_pad)
        #Second, dw is gradient with respect to w
        dw = np.zeros_like(w)
        #Third, db is gradient with respect to b
        db = np.zeros_like(b)
    
        #Now, we need to do work for backward of CNN, that means we have to make for loops same as conv_forward.
        #1. Input number index(input_idx)
        for input_idx in range(N):
            #2. Filter Index(filter_idx)
            for filter_idx in range(F):
                #3. Initialize looping for height first because filter moves direction right to left. (i)
                for i in range(output_height):
                    #4. Next, initialize looping for width (j)
                    for j in range(output_width):
                        #Defines dx, dw, db
                        #First, we keep add into dx about the result of upstream derivatives(dout) * w
                        dx[input_idx, i * SH:i * SH + WH,  j * SW:j * SW + WW, :] += dout[input_idx, i, j, filter_idx] * w[filter_idx, :, :, :]
                        #Second, we keep add into dw about the result of upstream derivatives(dout) * x_with_pad
                        dw[filter_idx, :, :, :] += dout[input_idx, i, j, filter_idx] * x_with_pad[input_idx, i * SH:i * SH + WH, j * SW:j * SW + WW, :]
                        #Third, db: is same as summation of dout
                        db[filter_idx] += dout[input_idx, i, j, filter_idx]
        
        #because of error ValueError: operands could not be broadcast together with shapes (2,6,8,3) (2,5,5,3)
        #We arranged it.
        dx = dx[:, (WH - 1) // 2: -(WH // 2), (WW - 1) // 2: -(WW // 2), :]
                        
    #Else if padding is valid, there is no padding so we need to make height and width of the output like
    #(m - k) / s + 1 which means in here, H - WH // SH + 1 and W - WW // SW + 1
    elif padding == 'valid':
        output_height = (H - WH) // SH + 1
        output_width = (W - WW) // SW + 1
        
        #Now, we need to initialize each dx, dw, db variables with same shape as x, w, b
        #Because we need the gradient with respect to each data index.
        #First, dx is gradient with respect to x
        dx = np.zeros_like(x)
        #Second, dw is gradient with respect to w
        dw = np.zeros_like(w)
        #Third, db is gradient with respect to b
        db = np.zeros_like(b)
    
        #Now, we need to do work for backward of CNN, that means we have to make for loops same as conv_forward.
        #1. Input number index(input_idx)
        for input_idx in range(N):
            #2. Filter Index(filter_idx)
            for filter_idx in range(F):
                #3. Initialize looping for height first because filter moves direction right to left. (i)
                for i in range(output_height):
                    #4. Next, initialize looping for width (j)
                    for j in range(output_width):
                        #Defines dx, dw, db
                        #First, dx: summation of upstream derivatives(dout) * w
                        dx[input_idx, i * SH:i * SH + WH,  j * SW:j * SW + WW, :] += dout[input_idx, i, j, filter_idx] * w[filter_idx, :, :, :]
                        #Second, dw: 
                        dw[filter_idx, :, :, :] += dout[input_idx, i, j, filter_idx] * x[input_idx, i * SH:i * SH + WH, j * SW:j * SW + WW, :]
                        #Third, db: this was just summation in conv_forward, so it is same as dout
                        db[filter_idx] += dout[input_idx, i, j, filter_idx]
        #Done getting variables (dx, dw, db)
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return dx, dw, db

def max_pool_forward(x, pool_param):
    """
    Computes the forward pass for a pooling layer.
    
    For your convenience, you only have to implement padding=valid.
    
    Inputs:
    - x: Input data, of shape (N, H, W, C)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The number of pixels between adjacent pooling regions, of shape (1, SH, SW, 1)

    Outputs:
    - out: Output data
    - cache: (x, pool_param)
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    pass
    #As conv_forward, initialize all variables first.
    N, H, W, C = x.shape
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    _, SH, SW, _= pool_param['stride']
    
    #For height and width of output,
    #This is same as 'valid' in conv_forward
    output_height = (H - PH) // SH + 1
    output_width = (W - PW) // SW + 1
    
    #Set 'out' variable 
    out = np.empty((N, output_height, output_width, C))
    
    #Making for loop for calculating output
    for i in range(output_height):
        for j in range(output_width):
            #Saves each outcome of index in mask for finding max inside of it.
            mask = x[:, i * SH: i *SH + PH, j * SW: j * SW + PW, :]
            #Find max using np.max with axis of height and width which are in 1, 2
            out[:, i, j, :] = np.max(mask, axis=(1, 2))
            
    #Done finding max pooling output.
            
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    Computes the backward pass for a max pooling layer.

    For your convenience, you only have to implement padding=valid.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in max_pool_forward.

    Outputs:
    - dx: Gradient with respect to x
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    pass
    #First, get all values from cache and initialize again
    x, pool_param = cache
    N, H, W, C = x.shape
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    _, SH, SW, _= pool_param['stride']
    
    #For height, width of output will be,
    output_height = (H - PH) // SH + 1
    output_width = (W - PW) // SW + 1
    
    #For dx, we need to have same shape as x
    dx = np.zeros_like(x)
    
    #Now, making for loop for calculating dx which will be mask * dout
    for i in range(output_height):
        for j in range(output_width):
            #As we made mask in max_pool_forward, we will make backward mask(same shape) as well here.
            mask_forward = x[:, i * SH: i *SH + PH, j * SW: j * SW + PW, :]
            #And then compares between two, for finding max
            #flag is for only the max value is True, and others are False
            flag = np.max(mask_forward, axis=(1, 2), keepdims=True) == mask_forward
            #And keep adding for dx
            dx[:, i * SH: i *SH + PH, j * SW: j * SW + PW, :] += flag * (dout[:, i, j, :])[:, None, None, :]
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return dx

def _rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def Test_conv_forward(num):
    """ Test conv_forward function """
    if num == 1:
        x_shape = (2, 4, 8, 3)
        w_shape = (2, 2, 4, 3)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.05, num=2)
        conv_param = {'stride': np.array([1,2,3,1]), 'padding': 'valid'}
        out, _ = conv_forward(x, w, b, conv_param)
        correct_out = np.array([[[[  5.12264676e-02,  -7.46786231e-02],
                                  [ -1.46819650e-03,   4.58694441e-02]],
                                 [[ -2.29811741e-01,   5.68244402e-01],
                                  [ -2.82506405e-01,   6.88792470e-01]]],
                                [[[ -5.10849950e-01,   1.21116743e+00],
                                  [ -5.63544614e-01,   1.33171550e+00]],
                                 [[ -7.91888159e-01,   1.85409045e+00],
                                  [ -8.44582823e-01,   1.97463852e+00]]]])
    else:
        x_shape = (2, 5, 5, 3)
        w_shape = (2, 2, 4, 3)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.05, num=2)
        conv_param = {'stride': np.array([1,3,2,1]), 'padding': 'same'}
        out, _ = conv_forward(x, w, b, conv_param)
        correct_out = np.array([[[[ -5.28344995e-04,  -9.72797373e-02],
                                  [  2.48150793e-02,  -4.31486506e-02],
                                  [ -4.44809367e-02,   3.35499072e-02]],
                                 [[ -2.01784949e-01,   5.34249607e-01],
                                  [ -3.12925889e-01,   7.29491646e-01],
                                  [ -2.82750250e-01,   3.50471227e-01]]],
                                [[[ -3.35956019e-01,   9.55269170e-01],
                                  [ -5.38086534e-01,   1.24458518e+00],
                                  [ -4.41596459e-01,   5.61752106e-01]],                             
                                 [[ -5.37212623e-01,   1.58679851e+00],
                                  [ -8.75827502e-01,   2.01722547e+00],
                                  [ -6.79865772e-01,   8.78673426e-01]]]])
        
    return _rel_error(out, correct_out)


def Test_conv_forward_IP(x):
    """ Test conv_forward function with image processing """
    w = np.zeros((2, 3, 3, 3))
    w[0, 1, 1, :] = [0.3, 0.6, 0.1]
    w[1, :, :, 2] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    b = np.array([0, 128])
    
    out, _ = conv_forward(x, w, b, {'stride': np.array([1,1,1,1]), 'padding': 'same'})
    plot_conv_images(x, out)
    return
    
def Test_max_pool_forward():   
    """ Test max_pool_forward function """
    x_shape = (2, 5, 5, 3)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    pool_param = {'pool_width': 2, 'pool_height': 3, 'stride': [1,2,4,1]}
    out, _ = max_pool_forward(x, pool_param)
    correct_out = np.array([[[[ 0.03288591,  0.03691275,  0.0409396 ]],
                             [[ 0.15369128,  0.15771812,  0.16174497]]],
                            [[[ 0.33489933,  0.33892617,  0.34295302]],
                             [[ 0.4557047,   0.45973154,  0.46375839]]]])
    return _rel_error(out, correct_out)

def _eval_numerical_gradient_array(f, x, df, h=1e-5):
    """ Evaluate a numeric gradient for a function """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        p = np.array(x)
        p[ix] = x[ix] + h
        pos = f(p)
        p[ix] = x[ix] - h
        neg = f(p)
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def Test_conv_backward(num):
    """ Test conv_backward function """
    if num == 1:
        x = np.random.randn(2, 4, 8, 3)
        w = np.random.randn(2, 2, 4, 3)
        b = np.random.randn(2,)
        conv_param = {'stride': np.array([1,2,3,1]), 'padding': 'valid'}
        dout = np.random.randn(2, 2, 2, 2)
    else:
        x = np.random.randn(2, 5, 5, 3)
        w = np.random.randn(2, 2, 4, 3)
        b = np.random.randn(2,)
        conv_param = {'stride': np.array([1,3,2,1]), 'padding': 'same'}
        dout = np.random.randn(2, 2, 3, 2)
    
    out, cache = conv_forward(x, w, b, conv_param)
    dx, dw, db = conv_backward(dout, cache)
    
    dx_num = _eval_numerical_gradient_array(lambda x: conv_forward(x, w, b, conv_param)[0], x, dout)
    dw_num = _eval_numerical_gradient_array(lambda w: conv_forward(x, w, b, conv_param)[0], w, dout)
    db_num = _eval_numerical_gradient_array(lambda b: conv_forward(x, w, b, conv_param)[0], b, dout)
    
    return (_rel_error(dx, dx_num), _rel_error(dw, dw_num), _rel_error(db, db_num))

def Test_max_pool_backward():
    """ Test max_pool_backward function """
    x = np.random.randn(2, 5, 5, 3)
    pool_param = {'pool_width': 2, 'pool_height': 3, 'stride': [1,2,4,1]}
    dout = np.random.randn(2, 2, 1, 3)
    
    out, cache = max_pool_forward(x, pool_param)
    dx = max_pool_backward(dout, cache)
    
    dx_num = _eval_numerical_gradient_array(lambda x: max_pool_forward(x, pool_param)[0], x, dout)
    
    return _rel_error(dx, dx_num)