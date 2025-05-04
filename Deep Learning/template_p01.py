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
    # Убедитесь, что decoder_hidden_state имеет размерность (n_features_dec, 1)
    assert decoder_hidden_state.shape[0] == W_mult.shape[0], "Size mismatch between decoder_hidden_state and W_mult"

    # Вычисляем оценки внимания
    attention_scores = W_mult.T @ decoder_hidden_state
    
    # Умножаем оценки на скрытые состояния кодера
    attention_weights = np.dot(attention_scores.T, encoder_hidden_states)  # (1, n_states)
    
    # Применяем softmax
    attention_weights = softmax(attention_weights)  # (1, n_states)
    
    # Вычисляем итоговый вектор внимания
    attention_vector = encoder_hidden_states @ attention_weights.T  # (n_features_enc, 1)
    
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
    # Применяем линейные преобразования
    encoder_transformed = W_add_enc @ encoder_hidden_states  # (n_features_int, n_states)
    decoder_transformed = W_add_dec @ decoder_hidden_state  # (n_features_int, 1)

    # Расширяем декодерное состояние для суммирования
    decoder_transformed_expanded = np.tile(decoder_transformed, (1, encoder_hidden_states.shape[1]))  # (n_features_int, n_states)

    # Суммируем результаты
    scores = v_add.T @ (encoder_transformed + decoder_transformed_expanded)  # (1, n_states)

    # Применяем активацию tanh
    scores = np.tanh(scores)

    # Применяем softmax к оценкам
    attention_weights = softmax(scores.T)  # (n_states, 1)

    # Вычисляем итоговый вектор внимания
    attention_vector = encoder_hidden_states @ attention_weights  # (n_features_enc, 1)

    return attention_vector
