from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params["W1"] = (
            np.random.randn(input_dim, hidden_dim) * weight_scale
        )  # self.params 让其他函数可以访问的变量
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params["b2"] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        N = X.shape[0]
        X = X.reshape(N, -1)  # 展平
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        hidden_layer = np.maximum(0, X.dot(W1) + b1)  # ReLU
        scores = hidden_layer.dot(W2) + b2  #  第二层
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                         #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # 用 softmax_loss 计算 loss 和 dscores
        loss, dscores = softmax_loss(scores, y)
        # 加上正则化
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        # 反向传播
        dhidden = dscores.dot(W2.T)
        dhidden[hidden_layer <= 0] = 0  # ReLU 的梯度
        grads["W2"] = hidden_layer.T.dot(dscores) + self.reg * W2
        grads["b2"] = np.sum(dscores, axis=0)
        grads["W1"] = X.T.dot(dhidden) + self.reg * W1
        grads["b1"] = np.sum(dhidden, axis=0)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        dims = [input_dim] + hidden_dims + [num_classes]

        # 循环 L-1 个隐藏层 + 1 个输出层
        for i in range(self.num_layers):
            # L 层的权重 W 连接第 L-1 层和第 L 层
            # W 形状: (dim_in, dim_out)
            # b 形状: (dim_out,)
            W_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)
            dim_in = dims[i]
            dim_out = dims[i + 1]

            # 初始化权重 W 和偏置 b
            self.params[W_key] = np.random.randn(dim_in, dim_out) * weight_scale
            self.params[b_key] = np.zeros(dim_out)

            # --- 批量归一化的修改 ---
            # 如果使用BN，为 *除最后一层外* 的所有层添加 gamma 和 beta
            if self.normalization == "batchnorm" and i < self.num_layers - 1:
                gamma_key = "gamma" + str(i + 1)
                beta_key = "beta" + str(i + 1)

                # gamma 初始化为 1
                self.params[gamma_key] = np.ones(dim_out)
                # beta 初始化为 0
                self.params[beta_key] = np.zeros(dim_out)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #               self.dropout_param                                       #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        X = X.reshape(X.shape[0], -1)

        current_input = X
        caches = {}  # 存储每一层的缓存，用于反向传播

        # --- 前向传播 ---

        # 循环 {affine - [bn] - relu} x (L - 1) 次
        for i in range(1, self.num_layers):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]

            if self.normalization == "batchnorm":
                # 使用我们创建的 affine_bn_relu 辅助函数
                gamma = self.params["gamma" + str(i)]
                beta = self.params["beta" + str(i)]
                # bn_params 是一个列表，我们需要传递正确的参数字典
                bn_param = self.bn_params[i - 1]

                current_input, cache = affine_bn_relu_forward(
                    current_input, W, b, gamma, beta, bn_param
                )
                caches[i] = cache
            else:
                # 原始的 affine - relu
                current_input, cache = affine_relu_forward(current_input, W, b)
                caches[i] = cache

        # 最后一层：只有 affine
        W_last = self.params["W" + str(self.num_layers)]
        b_last = self.params["b" + str(self.num_layers)]
        scores, cache_last = affine_forward(current_input, W_last, b_last)
        caches[self.num_layers] = cache_last
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ########################### BACKWARD PASS #################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        # 添加 L2 正则化损失
        total_W_sq_sum = 0
        for i in range(1, self.num_layers + 1):
            W = self.params["W" + str(i)]
            total_W_sq_sum += np.sum(W * W)
        loss += 0.5 * self.reg * total_W_sq_sum

        # 2. 最后一层的反向传播 (Affine)
        current_dout = dscores
        cache_last = caches[self.num_layers]

        dx, dw, db = affine_backward(current_dout, cache_last)

        grads["W" + str(self.num_layers)] = (
            dw + self.reg * self.params["W" + str(self.num_layers)]
        )
        grads["b" + str(self.num_layers)] = db

        current_dout = dx  # 更新上游梯度，准备进入循环

        # 3. 循环 L-1 个隐藏层的反向传播
        for i in range(self.num_layers - 1, 0, -1):
            cache = caches[i]

            if self.normalization == "batchnorm":
                # 使用 affine_bn_relu_backward 辅助函数
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(current_dout, cache)

                # 存储 BN 层的梯度
                grads["gamma" + str(i)] = dgamma
                grads["beta" + str(i)] = dbeta
            else:
                # 原始的 affine_relu_backward
                dx, dw, db = affine_relu_backward(current_dout, cache)

            # 存储仿射层的梯度 (W 和 b)
            grads["W" + str(i)] = dw + self.reg * self.params["W" + str(i)]
            grads["b" + str(i)] = db

            current_dout = dx  # 为下一次循环（更前一层）更新上游梯度

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True
