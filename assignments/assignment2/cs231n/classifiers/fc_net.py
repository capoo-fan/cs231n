from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


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
        dims = [input_dim] + hidden_dims + [num_classes] # 所有层的维度列表

        for i in range(self.num_layers): # i 从 0 到 L-1
            layer_num = i + 1
            
            # 初始化 W 和 b
            # W{i+1} 的形状是 (dims[i], dims[i+1])
            # b{i+1} 的形状是 (dims[i+1],)
            self.params[f'W{layer_num}'] = weight_scale * np.random.randn(dims[i], dims[i+1])
            self.params[f'b{layer_num}'] = np.zeros(dims[i+1])

            # 如果使用 BN 或 LN，并且不是最后一层，则初始化 gamma 和 beta
            if self.normalization is not None and i < self.num_layers - 1:
                # gamma 和 beta 的形状与 b 相同
                self.params[f'gamma{layer_num}'] = np.ones(dims[i+1])
                self.params[f'beta{layer_num}'] = np.zeros(dims[i+1])
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
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        current_input = X
        caches = {} # 存储每一层的缓存

        # --- 前向传播 ---
        # 循环 {affine - [bn/ln] - relu - [dropout]} x (L - 1) 次
        for i in range(1, self.num_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            
            cache_affine = None
            cache_norm = None
            cache_relu = None
            cache_dropout = None

            # 1. Affine
            current_input, cache_affine = affine_forward(current_input, W, b)
            
            # 2. Normalization
            if self.normalization == "batchnorm":
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
                bn_param = self.bn_params[i-1]
                current_input, cache_norm = batchnorm_forward(current_input, gamma, beta, bn_param)
            elif self.normalization == "layernorm":
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
                # ln_param 是一个空字典，但我们从 self.bn_params[i-1] 获取它以保持一致性
                ln_param = self.bn_params[i-1] 
                current_input, cache_norm = layernorm_forward(current_input, gamma, beta, ln_param)

            # 3. ReLU
            current_input, cache_relu = relu_forward(current_input)
            
            # 4. Dropout
            if self.use_dropout:
                current_input, cache_dropout = dropout_forward(current_input, self.dropout_param)

            # 存储所有缓存
            caches[i] = (cache_affine, cache_norm, cache_relu, cache_dropout)
            
        # 5. 最后一层 (第 L 层) - 只有 Affine
        W_last = self.params[f'W{self.num_layers}']
        b_last = self.params[f'b{self.num_layers}']
        scores, cache_last = affine_forward(current_input, W_last, b_last)
        caches[self.num_layers] = cache_last # 存储最后一层的缓存
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
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
        
        l2_reg_loss = 0.0
        for i in range(1, self.num_layers + 1):
            W = self.params[f'W{i}']
            l2_reg_loss += 0.5 * self.reg * np.sum(W * W)
        loss += l2_reg_loss

        # 2. 最后一层 (第 L 层) 的反向传播 (Affine)
        current_dout = dscores
        cache_affine_last = caches[self.num_layers]
        
        dx, dw, db = affine_backward(current_dout, cache_affine_last)
        
        grads[f'W{self.num_layers}'] = dw + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db
        
        current_dout = dx # 更新上游梯度，准备进入循环

        # 3. 循环 L-1 个隐藏层的反向传播
        for i in range(self.num_layers - 1, 0, -1):
            # 按照前向传播的相反顺序解包缓存
            (cache_affine, cache_norm, cache_relu, cache_dropout) = caches[i]
            
            # 4. Dropout backward
            if self.use_dropout:
                current_dout = dropout_backward(current_dout, cache_dropout)
            
            # 3. ReLU backward
            current_dout = relu_backward(current_dout, cache_relu)

            # 2. Normalization backward
            if self.normalization == "batchnorm":
                current_dout, dgamma, dbeta = batchnorm_backward_alt(current_dout, cache_norm)
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta
            elif self.normalization == "layernorm":
                current_dout, dgamma, dbeta = layernorm_backward(current_dout, cache_norm)
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta
                
            # 1. Affine backward
            dx, dw, db = affine_backward(current_dout, cache_affine)
            
            # 存储梯度
            grads[f'W{i}'] = dw + self.reg * self.params[f'W{i}']
            grads[f'b{i}'] = db
            
            # 为下一次循环（更前一层）更新上游梯度
            current_dout = dx
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
