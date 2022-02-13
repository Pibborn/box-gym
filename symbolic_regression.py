import numpy as np
from pysr import PySRRegressor
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("savedagents/" + "results", sep="\t")
    df = df.drop(df.columns[0], axis=1)
    df = df.reset_index()  # make sure indexes pair with number of rows

    pos1 = df.loc[:,"Position 1"][:-1]
    pos2 = df.loc[:,"Position 2"][:-1]
    den1 = df.loc[:,"Density 1"][:-1]
    den2 = df.loc[:,"Density 2"][:-1]
    size1 = df.loc[:,"Boxsize 1"][:]
    size2 = df.loc[:,"Boxsize 2"][:]

    left = np.array(list(zip(pos1, den1)))
    right = np.array(list(zip(pos2, den2)))

    input = np.array(list(zip(pos1, den1, den2)))
    output = np.array(pos2)

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