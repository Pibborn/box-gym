import pandas as pd
from gplearn.genetic import SymbolicRegressor
import numpy as np
import time

if __name__ == "__main__":
    file_name = "results_easy"
    file_name = "free_fall_20000"
    df = pd.read_csv(f"savedagents/extracted_data/{file_name}.csv")  # , sep="\t")
    df = df.drop(df.columns[0], axis=1)
    df = df.reset_index()  # make sure indexes pair with number of rows

    SCALE = 1
    BASKETBALL = 2
    mode = SCALE

    if mode == SCALE:
        # settings
        random_boxsizes = False
        actions = 1
        placed = 1
        boxes = actions + placed

        if random_boxsizes:
            number_of_columns = 3 * boxes - 1
            columns = [None for _ in range(number_of_columns)]
            for i in range(boxes - 1):
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
            columns[2 * boxes - 2] = df.loc[:, f"Density {boxes}"][:-1]

        # input = np.array(list(zip(columns[i] for i in range(number_of_columns))))
        # todo: don't hardcode it
        if boxes == 2:
            if random_boxsizes:
                input = np.array(list(zip(columns[0], columns[1], columns[2], columns[3], columns[4])))
            else:
                input = np.array(list(zip(columns[0], columns[1], columns[2])))
            # pos1, den1, den2, (size1, size2)
        elif boxes == 3:
            input = np.array(list(
                zip(columns[0], columns[1], columns[2], columns[3], columns[4], columns[5], columns[6], columns[7])))
            # pos1, pos2, den1, den2, den3, size1, size2, size3
        elif boxes == 4:
            input = np.array(list(zip(columns[0], columns[1], columns[2], columns[3], columns[4], columns[5],
                                      columns[6], columns[7], columns[8], columns[9], columns[10])))
        output = np.array(df.loc[:, f"Position {boxes}"][:-1])

    elif mode == BASKETBALL:
        """X = np.array(df[:, ["y-Position Ball", "Angle", "Velocity", "Radius", "Density", "x-Position Basket",
                          "y-Position Basket", "Radius Basket"]])
        X = np.transpose(X)
        y = df[:, "x-Position Ball"]"""
        input = np.array(df.loc[:, ["x-Position Ball", "y-Position Ball",  # "Force vector x", "Force vector y",
                                    "Radius", "Density", "x-Position Basket", "y-Position Basket", "Radius Basket"]])
        # input = np.transpose(input)
        y = np.array(df.loc[:, ["Force vector x", "Force vector y"]])
        output = np.array(df.loc[:, "Force vector x"])
        output = np.array(df.loc[:, "Force vector y"])

    est_gp = SymbolicRegressor(
        population_size=100000,  # todo: increase
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

    print()

    print(est_gp._program)

    print(est_gp.hall_of_fame)
    if mode == BASKETBALL:
        print(', '.join([f"x{i}: {column}]" for i, column in
                         enumerate(["x-Position Ball", "y-Position Ball",  # "Force vector x", "Force vector y",
                                    "Radius", "Density", "x-Position Basket", "y-Position Basket", "Radius Basket"])]))
