import numpy as np

def __cosine_similarity(r_i, r_j):
    norm_of_r_i = np.linalg.norm(r_i)
    norm_of_r_j = np.linalg.norm(r_j)
    
    norm_r_i_j = norm_of_r_i * norm_of_r_j
    
    if norm_r_i_j == 0:
        similarity_value = 0.0
    else:
        similarity_value = np.dot(r_i, r_j) / norm_r_i_j
        
    return similarity_value

def __pearson_correlation(r_i, r_j):
    pass #[TODO]

def get_similarity_matrix(normalized_ratings):

    similarity_metric = __cosine_similarity

    _n_items = normalized_ratings.shape[1]

    similarity_matrix = np.zeros((_n_items, _n_items), dtype='float')
    for i_idx_1 in xrange(_n_items):
        for i_idx_2 in xrange(_n_items):
            similarity_matrix[i_idx_1, i_idx_2] = similarity_metric(normalized_ratings[:, i_idx_1],
                                                                    normalized_ratings[:, i_idx_2])
                
    return similarity_matrix

def normalize_similarity_matrix(similarity_matrix):

    _n_items = similarity_matrix.shape[0]

    normalized_similarity_matrix = np.zeros(similarity_matrix.shape, dtype='float')
    for i_idx in xrange(_n_items):
        similarity_vector = similarity_matrix[i_idx]
        norm_of_similarity_vector = float(np.linalg.norm(similarity_vector))
        if norm_of_similarity_vector == 0:
            normalized_similarity_vector = similarity_vector
        else:
            normalized_similarity_vector = similarity_vector / norm_of_similarity_vector
        
        normalized_similarity_matrix[i_idx] = normalized_similarity_vector
    
    return normalized_similarity_matrix
