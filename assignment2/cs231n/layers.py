from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ## reshaping inputs ##
    row_dim = x.shape[0]
    col_dim = np.prod(x.shape[1:])
    x_reshaped = np.reshape(x, (row_dim, col_dim))
    
    ## computing the output ##
    out = (x_reshaped @ w) + b
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    row_dim = x.shape[0]
    col_dim = np.prod(x.shape[1:])
    x_reshaped = np.reshape(x, (row_dim, col_dim))
    dw = x_reshaped.T @ dout
    dx = np.reshape(dout @ w.T, x.shape)
    db = np.sum(dout, axis=0)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def affine_backward_conv(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #dout_reshaped = dout.reshape(-1)
    #x_reshaped = x.reshape(-1)
    #w_reshaped = w.reshape(-1)
    
    #print("x_reshaped.shape:", x_reshaped.shape)
    #print("dout_reshaped.shape:", dout_reshaped.shape)
    dw = (x_reshaped @ dout_reshaped)
    dx = (dout_reshaped @ w_reshaped)
    #dx = np.reshape(dout @ w_reshaped.T, x.shape)
    db = np.sum(dout, axis=0)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.maximum(x, 0)
    dx[dx!=0] = dout[dx!=0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0]
    f_exp_sums = 0 

    ## trick with logC:
    f = x - np.expand_dims(np.max(x, axis=1), axis=1)
    ## compute the denominator of zj:
    f_exp_sums = np.expand_dims(np.sum(np.exp(f), axis=1), axis=1)
    ## compute the probabilities
    p = np.exp(f) / f_exp_sums
    
    loss_matrix = p[range(p.shape[0]),y]
    loss_matrix = -np.log(loss_matrix+1e-10)
    loss = (np.sum(loss_matrix) / num_train)


    ## dx computation: (p[j] - (y[i] == j))* X[i]
    p_mask = np.zeros(p.shape)
    p_mask[np.arange(num_train), y] = 1
    dx = (p - p_mask) / num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)


    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    mb_mean = np.mean(x, axis=0)
    mb_var = np.var(x, axis=0)

    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        #mb_mean = np.mean(x, axis=0)
        #mb_var = np.sum((x - mb_mean)**2, axis=0)
        #mb_var = np.var(x, axis=0)

        running_mean = momentum * running_mean + (1-momentum) * mb_mean
        running_var = momentum * running_var + (1-momentum) * mb_var

        # applying normalization on mb input:
        x_norm = (x - mb_mean)/np.sqrt(mb_var + eps)

        # scaling and shifting using gamma, beta and storing in out
        out = gamma * x_norm + beta

        # storing original x, gamma and beta in cache:
        cache = (x, x_norm, gamma, beta, mb_mean, mb_var, eps)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    # storing original x, gamma and beta in cache:
    cache = (x, x_norm, gamma, beta, mb_mean, mb_var, eps)

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    x, x_norm, gamma, beta, mean, var, eps = cache

    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(x_norm * dout, axis = 0)

    # computing dx according to paper:
    N = x.shape[0]
    dx_norm = dout * gamma
    shifted_x = x-mean
    std_eps = np.sqrt(var + eps) 

    dvar = np.sum(dx_norm * shifted_x * -0.5 * ((var + eps)**(-1.5)), axis=0)
    dmean = np.sum(-dx_norm/std_eps, axis=0) + \
                      (dvar * ((np.sum(-2 * shifted_x, axis=0)) / N) )
    dx = (dx_norm / std_eps) + ((dvar * 2 * shifted_x) / N) + (dmean/N)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_norm, gamma, beta, mean, var, eps = cache
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_norm * dout, axis = 0)
    x_shifted = x - mean
    std = np.sqrt(var + eps)
    N = dout.shape[0]
    
    # computing dx - alternative
    dmean = 1/N * np.sum(dout, axis=0)
    dvar = 2/N * np.sum(x_shifted * dout, axis=0)
    dstddev = dvar/(2 * std)
    dx = gamma*((dout - dmean)*std - dstddev*(x_shifted))/std**2


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  
    
    ln_mean = np.mean(x.T, axis=0)
    ln_var = np.var(x.T, axis=0)
    # applying normalization on mb input:
    x_norm = (x.T - ln_mean)/np.sqrt(ln_var + eps)

    # scaling and shifting using gamma, beta and storing in out
    out = gamma * x_norm.T + beta
    # storing original x, gamma and beta in cache:
    
    #cache = (x.T, x_norm, gamma.reshape(-1,1), beta.reshape(-1,1), ln_mean, ln_var, eps)
    cache = (x.T, x_norm.T, gamma.reshape(-1,1), beta.reshape(-1,1), ln_mean, ln_var, eps)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    

    x, x_norm, gamma, beta, mean, var, eps = cache

    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(x_norm * dout, axis = 0)
    
    # computing dx according to paper:
    N = x.shape[0]
  
    dx_norm = dout.T * gamma
    shifted_x = x-mean
    std_eps = np.sqrt(var + eps) 

    dvar = np.sum(dx_norm * shifted_x * -0.5 * ((var + eps)**(-1.5)), axis=0)
    dmean = np.sum(-dx_norm/std_eps, axis=0) + \
                      (dvar * ((np.sum(-2 * shifted_x, axis=0)) / N) )
    dx = (dx_norm / std_eps) + ((dvar * 2 * shifted_x) / N) + (dmean/N)
    dx = dx.T
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask # drop!

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    
    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    
    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        dx = dout * mask
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx

def my_pad(x, pad, N, C, H, W):

    if (pad > 0):
      ## start by adding padding:
      pad_shape_row = (N, C, pad, W)
      pad_shape_col = (N, C, H+2*pad, pad)
      pad_row = np.zeros(pad_shape_row)
      pad_col = np.zeros(pad_shape_col)
      
      padded_x = np.append(x, pad_row, axis = 2)
      padded_x = np.append(pad_row, padded_x, axis = 2)
      padded_x = np.append(padded_x, pad_col, axis = 3)
      padded_x = np.append(pad_col, padded_x, axis = 3)
      ##################################################
    else:
      padded_x = x.copy()

    return padded_x

def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    
    #padded_x = my_pad(x, pad, N, C, H, W)
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    '''
    if (conv_param['pad']):
      ## start by adding padding:
      pad_shape_row = (N, C, pad, W)
      pad_shape_col = (N, C, H+2*pad, pad)
      pad_row = np.zeros(pad_shape_row)
      pad_col = np.zeros(pad_shape_col)
      
      padded_x = np.append(x, pad_row, axis = 2)
      padded_x = np.append(pad_row, padded_x, axis = 2)
      padded_x = np.append(padded_x, pad_col, axis = 3)
      padded_x = np.append(pad_col, padded_x, axis = 3)
      ##################################################
    else:
      padded_x = x.copy()'''
    
    #N, C, H, W = padded_x.shape

    ##### convolve: ##################################
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)

    out_shape = (N, F, H_out, W_out)
    out = np.zeros(out_shape)

    for n in range(N):
      for f in range(F):
        for i in range(H_out):
          for j in range(W_out):
            #selected_x = padded_x[n, :, i*stride:HH+i*stride, j*stride:WW+j*stride]
            #selected_x = selected_x.reshape(C*HH*WW)
            #curr_filter = w[f]
            #curr_filter = curr_filter.reshape(C*HH*WW)
            #conv = np.sum(selected_x * curr_filter) + b[f]
            conv = np.sum(padded_x[n, :, i*stride:HH+i*stride, j*stride:WW+j*stride] * w[f]) + b[f]
            out[n, f, i, j] = conv
    ###################################################


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache

    # dout should hold N * F * H_out * W_out derivatives
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)

    dx = np.zeros(x.shape)
    
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    ## pad the input x
    #padded_x = my_pad(x, pad, N, C, H, W)
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    ## init to 0 the padded dx
    padded_dx = np.zeros(padded_x.shape)
    

    # filter gradient dimensions: N, F, H_out, W_out
    
    # loop over all derivatives
    for n in range(N):
      for f in range(F):
        db[f] = np.sum(dout[:, f, :, :])
        for i in range(H_out):
          for j in range(W_out):
            ## get the filter map gradients:
            dw[f] += padded_x[n, :, i*stride:HH+i*stride, j*stride:WW+j*stride] * dout[n,f, i, j]
            padded_dx[n, :, i*stride:HH+i*stride, j*stride:WW+j*stride] += w[f] * dout[n,f, i, j]
            
    #remove the padding from dx to get to original x dimensions:
    dx = padded_dx[:, :, pad:H+pad, pad:W+pad]
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H, W = x.shape
    
    out_h = 1 + (H - pool_height)//stride
    out_w = 1 + (W - pool_height)//stride
    
    out = np.zeros((N, C, out_h, out_w))

    for n in range(N):
      for i in range(out_h):
        for j in range(out_w):
          switch = x[n, :, i*stride:i*stride + pool_height, j*stride:j*stride + pool_width]
          switch_max = np.max(switch, axis = (1, 2))
          out[n, :, i, j] = switch_max

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H, W = x.shape
    
    out_h = 1 + (H - pool_height)//stride
    out_w = 1 + (W - pool_height)//stride
    
    dx = np.zeros(x.shape)

    for n in range(N):
      for i in range(out_h):
        for j in range(out_w):
          for c in range(C):
            max_idx = \
            np.argmax(x[n, c, i*stride:i*stride + pool_height, j*stride:j*stride + pool_width])
            k, l = np.unravel_index(max_idx, (pool_height, pool_width))
            dx[n, c, i*stride+k, j*stride+l] = dout[n, c, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape

    # transpose to a channel-last notation (N, H, W, C) and then reshape it to 
    # norm over N*H*W for each C
    x = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    # transpose the output back to N, C, H, W
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    #xm, x_normm, gammam, betam, mb_meanm, mb_varm, epsm = cache[2]
    #print("gamma from cache after forward pass:", gammam)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape

    # transpose to a channel-last notation (N, H, W, C) and then reshape it to 
    # norm over N*H*W for each C
    dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)

    # transpose the output back to N, C, H, W
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the mean:
    N, C, H, W = x.shape

    x = x.reshape(N*G, C//G*H*W)
    
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    # applying normalization on mb input:
    x_norm = (x - mean)/np.sqrt(var + eps)
    
    # scaling and shifting using gamma, beta and storing in out
    x_norm = np.reshape(x_norm, (N, C, H, W)) 
    out = gamma*x_norm + beta

    #storing in cache:
    cache = x, x_norm, gamma, beta, mean, var, eps, G
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_norm, gamma, beta, mean, var, eps, G = cache
    N, C, H, W = dout.shape

    #dout = dout.reshape(N*G, C//G*H*W)
    dbeta = np.sum(dout, axis = (0, 2, 3), keepdims = True)
    dgamma = np.sum(x_norm * dout, axis = (0, 2, 3), keepdims = True)

    # computing dx according to paper:
    dx_norm = dout * gamma
    
    shifted_x = x-mean
    std_eps = np.sqrt(var + eps) 
    dx_norm = dx_norm.reshape(N*G, C//G*H*W)
    dvar = np.sum(dx_norm * shifted_x * -0.5 * ((var + eps)**(-1.5)), axis = 1, keepdims = True)
    shifted_x = shifted_x.reshape(N*G, C//G*H*W)
    dmean = np.sum(-dx_norm/std_eps, axis=1, keepdims = True) + \
                      (dvar * ((np.sum(-2 * shifted_x, axis=1, keepdims = True)) / (C//G*H*W)) )
    
    dx = (dx_norm / std_eps) + ((dvar * 2 * shifted_x) / (C//G*H*W)) + (dmean/(C//G*H*W))
    dx = dx.reshape((N, C, H, W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
