import os
import numpy as np
import pandas as pd


def create_train_test(input_file, data_type):
    """Create train and test sets from input file, for 2 possible data types

    Arguments:
        input_file {[string]} -- [path to csv data file]
        data_type {[string]} -- [the specified data type:
                                    'ratings': Movielens style ratings matrix
                                    'web_views': Time on page data]

    Raises:
        ValueError -- [the only supported data types are 'ratings','web_views']

    Returns:
        [array] -- [user IDs for each row of the ratings matrix]
        [array] -- [array of item IDs for each column of the rating matrix]
        TODO
        [sparse] -- [coo_matrix for training]
        [sparse] -- [coo_matrix for test]
    """

    if data_type == 'ratings':
        return _get_train_test_ratings(True, ',', input_file)
        # TODO: return sparse train and test sets
    elif data_type == 'web_views':
        # TODO: handle web_views case
        return _get_train_test_page_views(input_file)
    else:
        raise ValueError('unrecognized data type %s not supported' % data_type)


def _get_train_test_ratings(with_header, delimiter, input_file):
    if with_header:
        header_index = 0
    else:
        header_index = None
    ratings_df = pd.read_csv(input_file,
                             sep=delimiter,
                             header=header_index,
                             )
    ratings_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    del ratings_df['timestamp']

    (user_indicator, unordered_user, unique_user,
     max_user, nb_user) = _get_ordering_info(ratings_df, 'user_id')
    (item_indicator, unordered_item, unique_item,
     max_item, nb_item) = _get_ordering_info(ratings_df, 'item_id')

    if user_indicator or item_indicator:
        ordered_user_ids = _order_ids(unordered_user, unique_user,
                                      max_user, nb_user)
        ordered_item_ids = _order_ids(unordered_item, unique_item,
                                      max_item, nb_item)
        np_ratings = ratings_df['rating'].as_matrix()
        ratings = np.zeros((np_ratings.shape[0], 3), dtype=object)
        ratings[:, 0] = ordered_user_ids
        ratings[:, 1] = ordered_item_ids
        ratings[:, 2] = np_ratings
    else:
        ratings = ratings_df.as_matrix(['user_id', 'item_id', 'rating'])
        # deal with 1-based indices
        ratings[:, 0] -= 1
        ratings[:, 1] -= 1

    return ratings


def _get_ordering_info(ratings_df, specify_id):
    """This function performs some computations to tell if making\
       indexes for users of items is necessary or not

    Arguments:
        ratings_df {pandas data frame} -- Ratings dataset
        specify_id {string} -- 'user_id' or 'item_id' to be processed

    Returns:
        tuple -- contains boolean to indicate if the ids are unordered,\
                 the unordered ids, the ordered unique ids array, \
                 the maximum id number and the count of ids
    """
    if specify_id == 'user_id' or specify_id == 'item_id':
        # entity can be user or item
        unordered_entity = ratings_df[str(specify_id)].as_matrix()
        # ordered and unique ids
        unique_entity = np.unique(unordered_entity)
        nb_entity = unique_entity.shape[0]
        max_entity = unique_entity[-1]
        if max_entity != nb_entity:
            return (True, unordered_entity, unique_entity,
                    max_entity, nb_entity)
        else:
            return (False, unordered_entity, unique_entity,
                    max_entity, nb_entity)


def _order_ids(unordered_entity, unique_entity, max_entity, nb_entity):
    # make an array of 0-indexed unique entity ids
    zero_indexed = np.zeros(max_entity+1, dtype=int)
    zero_indexed[unique_entity] = np.arange(nb_entity)
    entity_r = zero_indexed[unordered_entity]
    return entity_r
