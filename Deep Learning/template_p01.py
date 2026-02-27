import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    # 数值稳定处理：减去行最大值，并用 np.newaxis 保持维度
    nice_vector = vector - np.max(vector, axis=1)[:, np.newaxis] 
    
    exp_vector = np.exp(nice_vector)
    # 求分母：沿着 axis=1 求和，并用 np.newaxis 保持维度
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # 1. 计算 Attention Scores (e_i = s^T W_mult h_i)
    # 步骤 1a: 计算 W_mult @ s
    temp_dec_proj = np.dot(W_mult, decoder_hidden_state) 
    
    # 步骤 1b: 计算 (temp_dec_proj)^T @ h
    attention_scores = np.dot(temp_dec_proj.T, encoder_hidden_states)
    
    # 2. 计算 Attention Weights (Softmax)
    attention_weights = softmax(attention_scores)
    
    # 3. 计算 Context Vector (加权求和)
    attention_vector = np.dot(encoder_hidden_states, attention_weights.T)
    
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # 1. 投影 Q 和 K 到注意力空间 (n_features_int)
    
    # 1a. 解码器状态投影 (W_add_dec @ s)
    # dec_proj 形状: (n_features_int, 1)
    dec_proj = np.dot(W_add_dec, decoder_hidden_state)
    
    # 1b. 编码器状态投影 (W_add_enc @ h)
    # enc_proj 形状: (n_features_int, n_states)
    enc_proj = np.dot(W_add_enc, encoder_hidden_states)
    
    # 2. 相加并应用激活函数 (tanh)
    # H_sum = W_add_enc h + W_add_dec s
    # dec_proj (n_features_int, 1) 会自动广播到 enc_proj (n_features_int, n_states)
    Sum_Proj = enc_proj + dec_proj
    H_tanh = np.tanh(Sum_Proj)
    
    # 3. 计算 Attention Scores (e_i = v_add^T @ H_tanh)
    # attention_scores 形状: (1, n_states)
    attention_scores = np.dot(v_add.T, H_tanh)
    
    # 4. 计算 Attention Weights (Softmax)
    # attention_weights 形状: (1, n_states)
    attention_weights = softmax(attention_scores)
    
    # 5. 计算 Context Vector (加权求和)
    # C = V @ alpha^T
    # attention_vector 形状: (n_features_enc, 1)
    attention_vector = np.dot(encoder_hidden_states, attention_weights.T)
    
    return attention_vector
