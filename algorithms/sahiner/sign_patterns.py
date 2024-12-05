import numpy as np


def check_if_already_exists(element_list, element):
    """
    The check_if_already_exists function checks if an element is in the list or not.

    Author(s): Arda Sahiner et al.

    @type: list
    @param element_list: list to search
    @param element: element to find
    @return: whether or not the element is in the list.
    """
    # check if element exists in element_list
    # where element is a numpy array
    return list(element) in element_list


def generate_sign_patterns(A, P, verbose=False):
    """
    The generate_sign_patterns function finds all the unique sign patterns for a data matrix.

    Author(s): Arda Sahiner et al.

    @param A: the data matrix to find sign patterns for
    @param P: the number of sign patterns
    @param verbose: the flag that indicates whether to print the number of unique sign patterns found.
    @return:
    """
    # generate sign patterns
    n, d = A.shape
    unique_sign_pattern_list = []  # sign patterns
    u_vector_list = []  # random vectors used to generate the sign paterns

    for i in range(P):
        # obtain a sign pattern
        u = np.random.normal(0, 1, (d, 1))  # sample u
        sampled_sign_pattern = (np.matmul(A, u) >= 0)[:, 0]

        # check whether that sign pattern has already been used
        if not check_if_already_exists(unique_sign_pattern_list, sampled_sign_pattern):
            unique_sign_pattern_list.append(list(sampled_sign_pattern))
            u_vector_list.append(u)

            if verbose and len(u_vector_list) % 10 == 0:
                print(i, 'generated', len(u_vector_list), 'unique sign patterns')

    if verbose:
        print("Number of unique sign patterns generated: " + str(len(unique_sign_pattern_list)))
    return unique_sign_pattern_list, u_vector_list
