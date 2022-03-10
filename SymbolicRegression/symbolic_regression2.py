import pandas as pd
from gplearn.genetic import SymbolicRegressor
import numpy as np

if __name__ == "__main__":
    random_densities = True
    df = pd.read_csv("savedagents/results7.csv")
    #df = df.drop(df.columns[0], axis=1)
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


    est_gp = SymbolicRegressor(
        population_size=50000,
        generations=30,
        stopping_criteria=0.01,
        p_crossover=0.7, p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.01,
        random_state=0
    )

    est_gp.fit(X=input, y=output)

    print(est_gp._program)

    print(est_gp.hall_of_fame)
