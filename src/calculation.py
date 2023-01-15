import math


def dot_product(vector1, vector2):
    new_vector = []
    total_sum = 0

    if len(vector1) == len(vector2):
        for index in range(len(vector1)):
            new_vector.append(vector1[index] * vector2[index])
        for index in range(len(new_vector)):
            total_sum += new_vector[index]
        return total_sum
    else:
        print("Vectors are not the same size")


def calculate_normal_idf(df, total_doc_count):
    return math.log(total_doc_count / df, 10)


def calculate_normal_idf_for_all(dfs, total_doc_count):
    idfs = []
    for index in range(len(dfs)):
        idfs.append(calculate_normal_idf(dfs[index], total_doc_count))
    return idfs


def calculate_normal_tfidf(tf, df, total_doc_count):
    return math.log(1 + tf, 10) * calculate_normal_idf(df, total_doc_count)


def vector_size(vector):
    total_sum = 0
    for index in range(len(vector)):
        total_sum += math.pow(vector[index], 2)
    return math.sqrt(total_sum)


def cos_vector(vector1, vector2):
    dot_result = dot_product(vector1, vector2)
    vector1_size = vector_size(vector1)
    vector2_size = vector_size(vector2)
    return dot_result / (vector1_size * vector2_size)


def make_vector_from_query(indexer, query_list):
    vector = [0] * indexer.get_term_count()
    for word in query_list:
        vector[indexer.get_term_index_from_list(word)] += 1
    return vector
