import numpy as np

def __get_rated_items(user_index, dat):
    """Get Rated Item Indexes for User with index [user_index]"""

    ratings_for_user = dat[user_index]
    rated_item_indexes_by_user = np.sort(np.where(ratings_for_user != 0)[0])

    return rated_item_indexes_by_user

def __get_similar_items(user_index, target_item_index, dat, similarity_matrix, k_most_similar):

    # Get rated items for user with index [user_index] except target item with index [target_item_index]
    rated_item_indexes_by_user = __get_rated_items(user_index, dat)
    rated_item_indexes_by_user = set(rated_item_indexes_by_user).difference(set([target_item_index]))
    rated_item_indexes_by_user = np.sort(list(rated_item_indexes_by_user))

    # Get similarity values between target item and rated items
    similarity_values = [similarity_matrix[target_item_index][rated_item_index] for rated_item_index in rated_item_indexes_by_user]
    similarity_values = np.array(similarity_values, dtype='float')

    # Sort rated items according to similarity values
    rated_item_sort_indexes = np.argsort(similarity_values)[::-1]
    rated_item_indexes_by_user = rated_item_indexes_by_user[rated_item_sort_indexes]
    similarity_values = similarity_values[rated_item_sort_indexes]

    # Get most [k_most_similar] similar items
    if len(rated_item_indexes_by_user) > k_most_similar:
        rated_item_indexes_by_user = rated_item_indexes_by_user[: k_most_similar]
        similarity_values = similarity_values[: k_most_similar]
    elif len(rated_item_indexes_by_user) < k_most_similar:
        print 'WARNING : User with index {} rated less then {} items except target item!'.format(user_index, k_most_similar)

    return rated_item_indexes_by_user, similarity_values


def predict_with_avg_distance(user_index, target_item_index, dat, similarity_matrix, k_most_similar, baseline_model):
        
    rated_item_indexes_by_user, similarity_values = __get_similar_items(user_index, target_item_index, dat, 
                                                                        similarity_matrix, k_most_similar)

    # Calculate prediction score
    abs_sum_similarities = np.sum(np.abs(similarity_values))
    baseline_predictor_score = baseline_model[user_index][target_item_index]

    prediction_score = 0
    for rated_item_index, sim_val in zip(rated_item_indexes_by_user, similarity_values):
        _rating_of_user = dat[user_index][rated_item_index]
        _score = float(sim_val * (_rating_of_user - baseline_predictor_score))
        prediction_score += _score

    prediction_score = (float(prediction_score) / float(abs_sum_similarities)) + float(baseline_predictor_score)
    
    return prediction_score

def predict_with_thresholding(user_index, target_item_index, dat, similarity_matrix, k_most_similar):

    rated_item_indexes_by_user, similarity_values = __get_similar_items(user_index, target_item_index, dat,
                                                                        similarity_matrix, k_most_similar)

    ## TODO: select [k_most_similar] nonnegative items. in this case nonnegative items <= [k_most_similar]
    select_positive_similarity_values = np.where(similarity_values >= 0)[0]
    rated_item_indexes_by_user = rated_item_indexes_by_user[select_positive_similarity_values]
    similarity_values = similarity_values[select_positive_similarity_values]

    # Calculate prediction score
    abs_sum_similarities = np.sum(np.abs(similarity_values))

    prediction_score = 0
    for rated_item_index, sim_val in zip(rated_item_indexes_by_user, similarity_values):
        _rating_of_user = dat[user_index][rated_item_index]
        _score = float(sim_val * _rating_of_user)
        prediction_score += _score

    prediction_score = float(prediction_score) / float(abs_sum_similarities)

    return prediction_score

def get_similar_items_using_similarity_matrix(target_item_index, similarity_matrix, top_k):
    """Get Similar items using similarity matrix"""

    n_items = similarity_matrix.shape[0]

    candidate_item_indexes = np.sort(list(set(range(n_items)).difference(set([target_item_index]))))
    similarity_values = similarity_matrix[target_item_index][candidate_item_indexes]

    candidate_item_sort_indexes = np.argsort(similarity_values)[::-1][: top_k]
    selected_item_indexes = candidate_item_indexes[candidate_item_sort_indexes]    
    selected_similarity_values = similarity_values[candidate_item_sort_indexes]

    return selected_item_indexes, selected_similarity_values

def get_dissimilar_items_using_similarity_matrix(target_item_index, similarity_matrix, top_k):
    """Get Similar items using dissimilarity matrix"""

    n_items = similarity_matrix.shape[0]

    candidate_item_indexes = np.sort(list(set(range(n_items)).difference(set([target_item_index]))))
    similarity_values = similarity_matrix[target_item_index][candidate_item_indexes]

    candidate_item_sort_indexes = np.argsort(similarity_values)[: top_k]
    selected_item_indexes = candidate_item_indexes[candidate_item_sort_indexes]
    selected_similarity_values = similarity_values[candidate_item_sort_indexes]

    return selected_item_indexes, selected_similarity_values

