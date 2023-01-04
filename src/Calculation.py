def dot_product(vector1, vector2):
    new_vector = []
    if len(vector1) == len(vector2):
        for index in range(len(vector1)):
            new_vector.append(vector1[index] * vector2[index])
        print(new_vector)
    else
        print("Vectors are not the same size")