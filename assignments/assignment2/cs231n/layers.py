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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    x_reshaped = x.reshape(x.shape[0], -1)
    # 2. 执行核心计算：out = x_reshaped * w + b
    out = x_reshaped.dot(w) + b
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
    num_examples = x.shape[0]
    flattened_x = x.reshape(num_examples, -1)  # 把每个样本展平成一维向量
    # 计算db
    db = np.sum(dout, axis=0)  # 对每个样本的梯度求和，得到每个输出维度的总梯度
    dw = flattened_x.T.dot(dout)  # 计算权重的梯度
    dx_flat = dout.dot(w.T)  # 计算输入的梯度（展平后的）
    dx = dx_flat.reshape(x.shape)  # 将梯度还原为输入的原始形状
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
    out = np.maximum(0, x)
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
    dx = dout * (x > 0)  # 计算 ReLU 的梯度
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
    score = x
    score -= np.max(
        score, axis=1, keepdims=True
    )  # axis=1表示 沿着行操作,keepdims=True 保持二维
    p = np.exp(score)
    p /= np.sum(p, axis=1, keepdims=True)  # 归一化
    logp = np.log(p)
    loss = -np.sum(
        logp[np.arange(x.shape[0]), y]
    )  # np.arange(X.shape[0]) 生成0~N-1的数组,用y索引出对应的值
    loss = loss / x.shape[0]

    dx = np.zeros_like(x)
    dscores = p.copy()
    dscores[np.arange(x.shape[0]), y] -= 1
    dx = dscores / x.shape[0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(
    x, gamma, beta, bn_param
):  # x是数据，gamma是缩放参数，beta是平移参数，bn_param是包含其他参数的字典
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
      - mode: 'train' or 'test'; required 训练或者测试模式
      - eps: Constant for numeric stability  数值稳定性的常数，防止方差为零
      - momentum: Constant for running mean / variance.  用于运行均值/方差的动量常数
      - running_mean: Array of shape (D,) giving running mean of features  特征的运行均值
      - running_var Array of shape (D,) giving running variance of features  特征的运行方差

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
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        sample_std = np.sqrt(sample_var + eps)
        x_normalized = (x - sample_mean) / sample_std
        out = gamma * x_normalized + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, sample_mean, sample_var, sample_std, x_normalized, gamma, eps)
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
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

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
    (x, sample_mean, sample_var, sample_std, x_normalized, gamma, eps) = cache
    N, D = x.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    dx_normalized = dout * gamma

    dstd_inv = 1.0 / sample_std
    dxc_dstd = -(x - sample_mean) * (sample_std**-2)  # -xc/sigma^2
    dstd = np.sum(dx_normalized * dxc_dstd, axis=0)  # 形状 (D,)

    dvar = dstd * (0.5 / sample_std)  # 形状 (D,)

    dxc_path1 = dx_normalized * dstd_inv  # 形状 (N, D)

    dxc_path2 = (2.0 / N) * (x - sample_mean) * dvar  # 形状 (N, D)

    # 将两个路径的梯度相加
    dxc = dxc_path1 + dxc_path2  # 形状 (N, D)

    dmu_path1 = -np.sum(dxc, axis=0)  # 形状 (D,)

    dmu = dmu_path1

    dx_path1 = dxc  # 形状 (N, D)

    dx_path2 = (1.0 / N) * dmu  # 形状 (N, D)

    dx = dx_path1 + dx_path2  # 形状 (N, D)
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
    (x, sample_mean, sample_var, sample_std, x_normalized, gamma, eps) = cache
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)

    dx_normalized = dout * gamma

    sum_dxn = np.sum(dx_normalized, axis=0)
    # sum(dx_hat * x_hat)
    sum_dxn_xhat = np.sum(dx_normalized * x_normalized, axis=0)

    # 4b. 组合成单行表达式 (如注释所提示)
    dx = (gamma / (N * sample_std)) * (
        (N * dx_normalized) - sum_dxn - (x_normalized * sum_dxn_xhat)
    )
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
    N, D = x.shape
    sample_mean = np.mean(x, axis=1, keepdims=True)  # 按行计算均值
    sample_var = np.var(x, axis=1, keepdims=True)  # 按行计算方差
    sample_std = np.sqrt(sample_var + eps)
    x_normalized = (x - sample_mean) / sample_std
    out = gamma * x_normalized + beta
    cache = (x, sample_mean, sample_var, sample_std, x_normalized, gamma, eps)
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
    (x, gamma, sample_mean, sample_var, sample_std, x_normalized, eps) = cache
    N, D = x.shape
    dbeta = np.sum(dout, axis=0)  # 形状 (D,)

    # dgamma = dL/dgamma = sum(dL/dy * x_hat) (沿 N 维求和)
    dgamma = np.sum(dout * x_normalized, axis=0)  # 形状 (D,)

    # 3. 计算 dL/d(x_hat) (与 BN 相同)
    # dL/dx_hat = dL/dy * gamma
    dx_normalized = dout * gamma  # 形状 (N, D)

    # 4. 计算 dx (使用 LN 版本的简化公式)
    # 这是从 batchnorm_backward_alt 修改而来的：
    # N -> D
    # axis=0 -> axis=1
    # sample_std 的形状现在是 (N, 1)，广播机制会自动处理

    # sum(dL/dx_hat) (沿 D 维求和)
    sum_dxn = np.sum(dx_normalized, axis=1, keepdims=True)  # 形状 (N, 1)

    # sum(dL/dx_hat * x_hat) (沿 D 维求和)
    sum_dxn_xhat = np.sum(
        dx_normalized * x_normalized, axis=1, keepdims=True
    )  # 形状 (N, 1)

    # 应用简化的反向传播公式
    # (gamma / (D * sample_std)) 是 (D,) / (N, 1) -> (N, D) (通过广播)
    # (D * dx_normalized) 是 (N, D)
    # (sum_dxn) 是 (N, 1)
    # (x_normalized * sum_dxn_xhat) 是 (N, D) * (N, 1) -> (N, D) (通过广播)
    dx = (gamma / (D * sample_std)) * (
        (D * dx_normalized) - sum_dxn - (x_normalized * sum_dxn_xhat)
    )
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
        mask = (
            np.random.rand(*x.shape) < p
        ) / p  # 每个元素有 p 的概率被保留，并进行缩放
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

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
        p = dropout_param["p"]
        dx = (dout * mask) / p
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


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
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))
    x_padded = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant", constant_values=0
    )
    for n in range(N):  # 遍历每张输入图片
        for f in range(F):  # 遍历每个滤波器
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    window = x_padded[n, :, h_start:h_end, w_start:w_end]  # 切片
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]
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
    x, w, b, conv_param = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    dx_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    x_padded = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant", constant_values=0
    )
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    window = x_padded[n, :, h_start:h_end, w_start:w_end]
                    dw[f] += window * dout[n, f, i, j]
                    db[f] += dout[n, f, i, j]
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += (
                        w[f] * dout[n, f, i, j]
                    )
    dx = dx_padded[:, :, pad : pad + H, pad : pad + W]
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
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_out = 1 + (H - pool_height) // stride  # 高度上的输出尺寸
    W_out = 1 + (W - pool_width) // stride  # 宽度上的输出尺寸
    out = np.zeros((N, C, H_out, W_out))
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    window = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(window)
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
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    window = x[n, c, h_start:h_end, w_start:w_end]
                    max_val = np.max(window)
                    mask = window == max_val
                    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]
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
    N, C, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
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
    N, C, H, W = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
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
    # 1. 获取输入维度
    N, C, H, W = x.shape

    # 2. 关键步骤：将输入 (N, C, H, W) 重塑为 (N, G, D_group)
    #    其中 D_group = (C/G) * H * W，这是我们要归一化的维度
    #    (N, C, H, W) -> (N, G, C//G, H, W)  (在概念上分组)
    #    (N, G, C//G, H, W) -> (N * G, C//G * H * W) (为 2D 归一化做准备)
    #    我们选择 (N*G, -1) 是为了方便调用普通的 norm 函数，
    #    但从数学上讲，(N, G, -1) 更清晰。我们这里用 (N, G, -1) 来计算。
    x_grouped = x.reshape(N, G, -1)  # 形状变为 (N, G, (C//G)*H*W)
    # 3. 像层归一化 (Layer Norm) 一样，计算均值和方差
    #    但我们是在最后一个维度 (D_group) 上计算
    #    为每个样本 (N) 和每个组 (G) 独立计算
    mean = np.mean(x_grouped, axis=2, keepdims=True)  # 形状 (N, G, 1)
    var = np.var(x_grouped, axis=2, keepdims=True)  # 形状 (N, G, 1)
    std = np.sqrt(var + eps)  # 形状 (N, G, 1)
    # 4. 执行归一化
    x_normalized_grouped = (x_grouped - mean) / std  # 形状 (N, G, D_group)
    # 5. 将形状恢复为 (N, C, H, W)
    x_normalized = x_normalized_grouped.reshape(N, C, H, W)
    # 6. 应用逐通道的缩放 (gamma) 和平移 (beta)
    #    gamma 和 beta 的形状是 (1, C, 1, 1)，它们会自动广播
    out = x_normalized * gamma + beta
    # 7. 存储反向传播所需的值
    cache = (x, G, mean, var, std, x_normalized_grouped, gamma, eps)
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
    # 1. 从缓存解压前向传播时保存的变量
    # (假设 cache 是按照我们前一个回答中 `spatial_groupnorm_forward` 的方式保存的)
    x, G, mean, var, std, x_normalized_grouped, gamma, eps = cache

    # 2. 获取维度
    N, C, H, W = dout.shape
    D_group = (C // G) * H * W

    # 3. 步骤 1：反向传播 gamma 和 beta
    # dbeta 和 dgamma 的形状是 (1, C, 1, 1)
    # 我们沿着 N, H, W 维度求和
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    # x_normalized = x_normalized_grouped.reshape(N, C, H, W)
    x_normalized = x_normalized_grouped.reshape(N, C, H, W)
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3), keepdims=True)

    # 上游梯度 dx_normalized (形状 (N, C, H, W))
    dx_normalized = dout * gamma

    # 4. 步骤 2：反向传播 Reshape
    # (N, C, H, W) -> (N, G, D_group)
    dx_normalized_grouped = dx_normalized.reshape(N, G, -1)

    # 5. 步骤 3：反向传播归一化 (复制 layernorm_backward 的逻辑)

    # 5a. sum(dL/dx_hat) (沿 D_group 维求和)
    sum_dxn = np.sum(dx_normalized_grouped, axis=2, keepdims=True)  # 形状 (N, G, 1)

    # 5b. sum(dL/dx_hat * x_hat) (沿 D_group 维求和)
    sum_dxn_xhat = np.sum(
        dx_normalized_grouped * x_normalized_grouped, axis=2, keepdims=True
    )  # 形状 (N, G, 1)

    # 5c. 应用简化的 LN 反向传播公式 (适应 SGN)
    # (1 / (D_group * std)) 的形状是 (N, G, 1)
    # (D_group * dx_normalized_grouped) 是 (N, G, D_group)
    # (sum_dxn) 是 (N, G, 1)
    # (x_normalized_grouped * sum_dxn_xhat) 是 (N, G, D_group)
    dx_grouped = (1.0 / (D_group * std)) * (
        (D_group * dx_normalized_grouped)
        - sum_dxn
        - (x_normalized_grouped * sum_dxn_xhat)
    )

    # 6. 步骤 4：反向传播 Reshape
    # (N, G, D_group) -> (N, C, H, W)
    dx = dx_grouped.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
