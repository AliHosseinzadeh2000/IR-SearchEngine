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


def vector_size(vector):
    total_sum = 0
    for index in range(len(vector)):
        total_sum += math.pow(vector[index], 2)
    return math.sqrt(total_sum)


def cos_vector(vector1, vector2):
    dot_result = dot_product(vector1, vector2)
    vector1_size = vector1_size(vector1)
    vector2_size = vector_size(vector2)
    return dot_result / (vector1_size * vector2_size)
