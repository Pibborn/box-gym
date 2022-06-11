import time

import numpy as np
from pysr import PySRRegressor
import pandas as pd

if __name__ == "__main__":
    file_name = "result10000"
    df = pd.read_csv(f"savedagents/extracted_data/{file_name}.csv")  # , sep="\t")
    df = df.drop(df.columns[0], axis=1)
    df = df.reset_index()  # make sure indexes pair with number of rows

    SCALE = 1
    BASKETBALL = 2
    mode = 1
    if mode == SCALE:
        # settings
        random_boxsizes = True
        actions = 1
        placed = 2
        boxes = actions + placed

        """pos1 = df.loc[:,"Position 1"][:-1]
        pos2 = df.loc[:,"Position 2"][:-1]
        pos3 = df.loc[:,"Position 3"][:-1]
        den1 = df.loc[:,"Density 1"][:-1]
        den2 = df.loc[:,"Density 2"][:-1]
        den3 = df.loc[:,"Density 3"][:-1]
        size1 = df.loc[:,"Boxsize 1"][:]
        size2 = df.loc[:,"Boxsize 2"][:]
        size3 = df.loc[:,"Boxsize 3"][:]
        
        if not random_boxsizes:
            left = np.array(list(zip(pos1, den1)))
            right = np.array(list(zip(pos2, den2)))

            input = np.array(list(zip(pos1, den1, den2)))
            output = np.array(pos2)
        else:
            left = np.array(list(zip(pos1, den1, size1)))
            right = np.array(list(zip(pos2, den2, size2)))

            #input = np.array(list(zip(pos1, den1, den2, size1, size2)))
            #output = np.array(pos2)
            input = np.array(list(zip(pos1, pos2, den1, den2, den3, size1, size2, size3)))
            output = np.array(pos3)"""

        if random_boxsizes:
            number_of_columns = 3 * boxes - 1
            columns = [None for _ in range(number_of_columns)]
            for i in range(boxes-1):
                columns[i] = df.loc[:, f"Position {i + 1}"][:-1]
                columns[boxes + i - 1] = df.loc[:, f"Density {i + 1}"][:-1]
                columns[2 * boxes + i - 1] = df.loc[:, f"Boxsize {i + 1}"][:]
            columns[2 * boxes - 2] = df.loc[:, f"Density {boxes}"][:-1]
            columns[3 * boxes - 2] = df.loc[:, f"Boxsize {boxes}"][:]

        else:
            number_of_columns = 2 * boxes - 1
            columns = [None for _ in range(number_of_columns)]
            for i in range(boxes - 1):
                columns[i] = df.loc[:, f"Position {i + 1}"][:-1]
                columns[boxes + i - 1] = df.loc[:, f"Density {i + 1}"][:-1]
            columns[2 * boxes - 1] = df.loc[:, f"Density {boxes}"][:-1]

        input = np.array([zip(column for column in columns)])
        output = np.array(df.loc[:, f"Position {boxes}"][:-1])
    elif mode == BASKETBALL:
        pass

    model = PySRRegressor(
        niterations=5,
        binary_operators=["+", "*"],
        unary_operators=["cos",
                         "exp",
                         "sin",
                         "inv(x)=1/x",
                         ],
        model_selection="best",
        #loss="(X, y) = (X - y)^2",
    )

    #model.fit(left, right)
    model.fit(input, output)
    print(model)