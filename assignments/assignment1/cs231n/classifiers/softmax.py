from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W) # 取得分数

        # compute the probabilities in numerically stable way
        scores -= np.max(scores) # 防止 exp 数值过大
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        dscores = p.copy()
        dscores[y[i]] -= 1
        for j in range(num_classes):
            dW[:, j] += X[i] * dscores[j] # 对于同一个第j类,累加所有样本的贡献
   
    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    #############################################################################
    # TODO:        
                                                             #
    # Compute the gradient of the loss function and store it dW.          
    # 计算损失函数的梯度,将其存储为 dW
    # Rather that first computing the loss and then computing the derivative,   #
    #  在计算损失的同时计算梯度
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                 
    # 在计算损失的同时计算梯度
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    score=X.dot(W)
    score-=np.max(score,axis=1,keepdims=True) # axis=1表示 沿着行操作,keepdims=True 保持二维
    p=np.exp(score)
    p/=np.sum(p,axis=1,keepdims=True) # 归一化
    logp=np.log(p)
    loss=-np.sum(logp[np.arange(X.shape[0]),y]) # np.arange(X.shape[0]) 生成0~N-1的数组,用y索引出对应的值
    loss=loss/X.shape[0]+reg*np.sum(W*W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    dW=np.zeros_like(W)
    dscores=p.copy()
    dscores[np.arange(X.shape[0]),y]-=1
    dW=X.T.dot(dscores)
    dW=dW/X.shape[0]+2*reg*W
    return loss, dW
