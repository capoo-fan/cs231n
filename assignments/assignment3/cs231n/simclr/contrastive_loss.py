import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    product= torch.sum(z_i * z_j)
    norm_i = torch.linalg.norm(z_i)
    norm_j = torch.linalg.norm(z_j)
    norm_dot_product = product / (norm_i * norm_j)  
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # --- 第一部分：计算 l(k, k+N) ---
        # 1. 分子：正样本对的指数相似度
        # sim_pos = exp(sim(z_k, z_k_N) / tau)
        sim_val = sim(z_k, z_k_N) / tau
        numerator = torch.exp(sim_val)

        # 2. 分母：z_k 与所有其他样本（除了它自己）的指数相似度之和
        denominator = 0
        for i in range(2 * N):
            if i != k:  # 排除自己 (k != i)
                sim_neg = sim(z_k, out[i]) / tau
                denominator += torch.exp(sim_neg)

        # 3. 累加损失: -log(分子 / 分母)
        total_loss += -torch.log(numerator / denominator)

        # --- 第二部分：计算 l(k+N, k) (对称项) ---
        # 1. 分子：正样本对的指数相似度 (与上面相同，因为 sim(u,v) = sim(v,u))
        # 但为了逻辑清晰，我们重新写一遍，锚点变成了 z_k_N
        sim_val_2 = sim(z_k_N, z_k) / tau
        numerator_2 = torch.exp(sim_val_2)

        # 2. 分母：z_k_N 与所有其他样本（除了它自己）的指数相似度之和
        denominator_2 = 0
        for i in range(2 * N):
            if i != k + N:  # 排除自己
                sim_neg = sim(z_k_N, out[i]) / tau
                denominator_2 += torch.exp(sim_neg)

        # 3. 累加损失
        total_loss += -torch.log(numerator_2 / denominator_2)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    product= torch.sum(out_left * out_right, dim=1)
    norm_left = torch.linalg.norm(out_left, dim=1)
    norm_right = torch.linalg.norm(out_right, dim=1)
    pos_pairs = product / (norm_left * norm_right)
    pos_pairs = pos_pairs.view(-1,1)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    # 1. 计算每个向量的模长 (L2 Norm)
    # out shape: (2N, D) -> norms shape: (2N, 1)
    # keepdim=True 是为了方便后续广播除法
    norms = torch.linalg.norm(out, dim=1, keepdim=True)

    # 2. 归一化向量
    # 这一步相当于把公式中的分母 ||z_i|| * ||z_j|| 提前处理了
    # 归一化后，向量的点积就直接等于余弦相似度
    out_normalized = out / norms

    # 3. 矩阵乘法计算所有两两点积
    # (2N, D) @ (D, 2N) -> (2N, 2N)
    sim_matrix = torch.matmul(out_normalized, out_normalized.t())

    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    # 计算所有两两相似度的指数值
    exponential = torch.exp(sim_matrix / tau)
    
    # This binary mask zeros out terms where k=i.
    # 创建一个掩码，把对角线（自己和自己的相似度）去掉，因为公式里要求 k != i
    mask = (
        (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool())
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]

    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    # 对每一行求和，得到分母
    denom = exponential.sum(dim=1, keepdim=True)

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    positive_pairs = sim_positive_pairs(out_left, out_right)  # [2*N]

    # Step 3: Compute the numerator value for all augmented samples.
    numerator = torch.exp(positive_pairs / tau).unsqueeze(1)  # [2*N, 1]

    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    # 公式： -log(分子 / 分母)
    # 最后求 mean() 是因为我们要计算整个 batch 的平均损失
    loss = -torch.log(numerator / denom).mean()
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))