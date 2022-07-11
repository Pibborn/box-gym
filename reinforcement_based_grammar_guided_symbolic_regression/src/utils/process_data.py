import numpy as np
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    identity_matrix = np.eye(num_classes, dtype='uint8')
    return identity_matrix[y]

def insert_variable_names (equation, feature_names):
    for i in range(len(feature_names)):
        equation=equation.replace(f'x[:,{i}]', str(feature_names[i]))
    return equation
