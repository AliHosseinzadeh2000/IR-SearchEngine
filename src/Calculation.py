import math


def dot_product(vector1, vector2):
    new_vector = []
    if len(vector1) == len(vector2):
        for index in range(len(vector1)):
            new_vector.append(vector1[index] * vector2[index])
        print(new_vector)
    else:
        print("Vectors are not the same size")


def calculate_normal_idf(df, total_doc_count):
    return math.log(total_doc_count / df, 10)


def calculate_normal_tfidf(tf, df):
    return math.log(1 + tf, 10) * calculate_normal_idf(df)
