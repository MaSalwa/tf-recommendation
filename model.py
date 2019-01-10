import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# ratio of train to test set size
TEST_SET_RATIO = 10


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
        [sparse] -- [scipy coo_matrix for training]
        [sparse] -- [scipy coo_matrix for test]
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
    """This function process ratings data to extract user and item
        ids, sparse train and test sets

    Arguments:
        with_header {boolean} -- True if file has header
        delimiter {delimiters} -- delimiter used in the csv file
        input_file {string} -- ratings csv file full path

    Returns:
        list -- contains 0-indexed user and item ids array and
                COO format sparse train and test matrices
    """
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
    sparse_train, sparse_test = _get_sparse_train_test(ratings,
                                                       nb_user, nb_item)
    return ratings[:, 0], ratings[:, 1], sparse_train, sparse_test


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
    """This function is called only if making 0-indexed ids is needed.\
       It creates an array that contains ordered 0-indexed unique ids\
       mapping to the original unordered ids.

    Arguments:
        unordered_entity {numpy matrix} -- the original unordered ids
        unique_entity {numpy array} -- unique ordered ids
        max_entity {int} -- the biggest id value in the ordered unique array
        nb_entity {int} -- the length of the ids array

    Returns:
        [numpy array] -- 0 indexed indices corresponding to original ones
    """
    # make an array of 0-indexed unique entity ids
    zero_indexed = np.zeros(max_entity+1, dtype=int)
    zero_indexed[unique_entity] = np.arange(nb_entity)
    entity_r = zero_indexed[unordered_entity]
    return entity_r


def _get_sparse_train_test(ratings, nb_user, nb_item):
    """Divide ratings data into train and test sets
       and transform them into scipy coo sparse matrices

    Arguments:
        ratings {numpy matrix} -- processed 0-indexed user
                                  and item ids ratings dataset
        nb_user {int} -- number of users
        nb_item {int} -- number of items

    Returns:
        list --  sparse scipy coo matrix test and train
    """
    l = len(ratings)
    test_set_size = l / TEST_SET_RATIO
    # pick random test set of entries, in ascending order
    test_set_idx = sorted(np.random.choice(xrange(l),
                          size=test_set_size, replace=False))
    # divide ratings into train and test sets
    ratings_test = ratings[test_set_idx]
    ratings_train = np.delete(ratings, test_set_idx, axis=0)
    sparse_test = _create_coo_matrix(ratings_test, nb_user, nb_item)
    sparse_train = _create_coo_matrix(ratings_train, nb_user, nb_item)
    return sparse_test, sparse_train


def _create_coo_matrix(ratings_part, nb_user, nb_item):
    """takes rating partitioned in either train or test,
        unzips them into user, item and ratings arrays
        and creates scipy coo matrics from them

    Arguments:
        ratings_part {numpy array} -- partioned ratings data
        nb_user {int} -- number of users
        nb_item {int} -- number of items

    Returns:
        scipy coo sparse -- matrix of ratings partioned (train or test)
    """

    # "part" is the partition, train or test
    # unzipping values to get the user, item and ratings seperately
    user_p, item_p, rating_p = zip(*ratings_part)
    sparse_part = coo_matrix((rating_p, (user_p, item_p)),
                             shape=(nb_user, nb_item))
    return sparse_part
