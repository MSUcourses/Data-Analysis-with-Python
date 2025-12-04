import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
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
    # Считаем промежуточное произведение W_mult * h_i
    transformed_encoder = W_mult.dot(encoder_hidden_states)  # shape: (n_features_dec, n_states)
    # Скаляное произведение с состоянием декодера
    attention_scores = decoder_hidden_state.T.dot(transformed_encoder)  # shape: (1, n_states)
    # Применяем softmax для получения весов
    attention_weights = softmax(attention_scores)
    # Итоговый attention vector
    attention_vector = attention_weights.dot(encoder_hidden_states.T).T  # shape: (n_features_enc, 1)
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
    n_states = encoder_hidden_states.shape[1]
    attention_scores = np.zeros((1, n_states))
    
    # Вычисляем e_i для каждого состояния энкодера
    for i in range(n_states):
        h_i = encoder_hidden_states[:, i][:, None]  # вектор состояния i
        e_i = np.dot(v_add.T, np.tanh(np.dot(W_add_enc, h_i) + np.dot(W_add_dec, decoder_hidden_state)))
        attention_scores[0, i] = e_i
    
    # Применяем softmax для получения весов
    attention_weights = softmax(attention_scores)
    
    # Итоговый attention vector: взвешенная сумма состояний энкодера
    attention_vector = attention_weights.dot(encoder_hidden_states.T).T
    
    return attention_vector


