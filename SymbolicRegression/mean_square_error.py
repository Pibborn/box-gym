import numpy as np
from pysr import PySRRegressor
import pandas as pd
import statsmodels.formula.api as sm

if __name__ == "__main__":
    random_densities = True

    path = "results7"
    df = pd.read_csv(f"../savedagents/{path}.csv")
    df = df.reset_index()  # make sure indexes pair with number of rows

    pos1 = df.loc[:, "Position 1"][:-1]
    pos2 = df.loc[:, "Position 2"][:-1]
    den1 = df.loc[:, "Density 1"][:-1]
    den2 = df.loc[:, "Density 2"][:-1]
    size1 = df.loc[:, "Boxsize 1"][:]
    size2 = df.loc[:, "Boxsize 2"][:]

    if not random_densities:
        left = np.array(list(zip(pos1, den1)))
        right = np.array(list(zip(pos2, den2)))

        input = np.array(list(zip(pos1, den1, den2)))
        output = np.array(pos2)
    else:
        left = np.array(list(zip(pos1, den1, size1)))
        right = np.array(list(zip(pos2, den2, size2)))

        input = np.array(list(zip(pos1, den1, den2, size1, size2)))
        output = np.array(pos2)

    goal_function = "(-pos1 * den1 * (2 * size1) ** 3) / (den2 * (2 * size2) ** 3)"
    observed_functions = ["((((size1 *  (pos1 - 1.4712403)) * 1 / ((-1.1480063 * (size2 + 0.07193513)) * size2)) * size1) - 0.3220555)",
                          ]

    errors = {}
    for f in observed_functions:
        mse = np.nansum(np.array((pos2 - eval(f)) ** 2)) / len(pos1)
        errors[f] = mse
        print(f"Mean squared error for function 'pos2 = {f}': {mse}")

    optimal_mse = np.nansum((pos2 - eval(goal_function)) ** 2) / len(pos1)
    print(f"Mean squared error for goal function 'pos2 = {goal_function}': {optimal_mse}")

    #print(errors.sorted(lambda x: x))



    """df = pd.DataFrame({"pos1": pos1, "pos2": pos2, "den1": den1, "den2": den2})
    result = sm.ols(formula="pos1 ~ pos2 * den2 / den1", data=df).fit()
    print(result.params)
    print(result.summary())"""
