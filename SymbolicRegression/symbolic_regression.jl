using Pkg
using SymbolicRegression
using CSV
using DataFrames

SCALE = 1
BASKETBALL = 2
FREEFALL = 3

function symbolic_regression(file_name="free_fall_1000", mode=FREEFALL)
#function symbolic_regression(file_name="Basketball_random_gravity", mode=BASKETBALL)
#function symbolic_regression(file_name="results8", mode=SCALE)
    df = DataFrame(CSV.File("savedagents/extracted_data/$file_name.csv"))
    df = select!(df, Not("Column1")) # cut the first column

    if (mode == SCALE)
        random_boxsizes = true
        actions = 1
        placed = 1
        boxes = actions + placed

        if (random_boxsizes)
            number_of_columns = 3 * boxes - 1
            columns = Array{String}(undef, number_of_columns)
            for i in 1:boxes-1
                columns[i] = "Position $i"
                columns[boxes + i - 1] = "Density $i"
                columns[2 * boxes + i - 1] = "Boxsize $i"
            end
            columns[2 * boxes - 1] = "Density $boxes"
            columns[3 * boxes - 1] = "Boxsize $boxes"
            #println(columns)
            #X = Matrix(df[:, ["Position 1", "Position 2", "Density 1", "Density 2", "Density 3", "Boxsize 1", "Boxsize 2", "Boxsize 3"]])
        else
            number_of_columns = 2 * boxes - 1
            columns = Array{String}(undef, number_of_columns)
            for i in 1:boxes-1
                columns[i] = "Position $i"
                columns[boxes + i - 1] = "Density $i"
            end
            columns[2 * boxes - 1] = "Density $boxes"
            #X = Matrix(df[:, ["Position 1", "Density 1", "Density 2"]])
        end
        X = Matrix(df[:, columns])
        X = transpose(X)
        print(columns)
        y = df."Position 2" # todo: don't hardcode it
    elseif (mode==BASKETBALL)
       """df = pd.DataFrame({'x-Position Ball': pd.Series(dtype='float'),
                           'y-Position Ball': pd.Series(dtype='float'),
                           'Angle': pd.Series(dtype='float'),
                           'Velocity': pd.Series(dtype='float'),
                           'Radius': pd.Series(dtype='float'),
                           'Density': pd.Series(dtype='float'),
                           'x-Position Basket': pd.Series(dtype='float'),
                           'y-Position Basket': pd.Series(dtype='float'),
                           'Radius Basket': pd.Series(dtype='float'),
                           })"""
       # old version
       #X = Matrix(df[:, ["y-Position Ball", "Angle", "Velocity", "Radius", "Density", "x-Position Basket", "y-Position Basket", "Radius Basket"]])

       # new version
       #X = Matrix(df[:, ["x-Position Ball", "y-Position Ball", "Radius", "Density", "x-Position Basket", "y-Position Basket", "Radius Basket"]])
       X = Matrix(df[:, ["x-Position Start", "y-Position Start", "Force vector x", "Force vector y", "Radius", "Density", "x-Position End", "y-Position End", "Velocity x", "Velocity y", "Gravity", "Time"]])
       X = Matrix(df[:, ["x-Position Start", "y-Position Start", "Force vector x", "Force vector y", "Radius", "Density", "x-Position End", "y-Position End", "Velocity x", "Velocity y", "Gravity"]])
       X = Matrix(df[:, ["Force vector y", "Velocity y", "Gravity"]])
       X = Matrix(df[:, ["Force vector x", "Force vector y", "Velocity x", "Velocity y", "Gravity"]])
       X = Matrix(df[:, ["Force vector x", "Velocity x", "Velocity y", "Gravity", "y-Position Start", "y-Position End"]])
       X = transpose(X)

       # old
       # y = df."x-Position Ball"
       # new
       # y = df[:, ["Force vector x", "Force vector y"]]
       y = df."Force vector y"
       # vt = v0 - g * t --> t = (vt - v0) / g
       #y = df."Velocity y"

    elseif (mode == FREEFALL)
        X = Matrix(df[:, ["Distance", "Velocity", "Radius", "Density", "Gravity", "Time"]])
        X = Matrix(df[:, ["Gravity", "Time"]]) #, "Radius"]])
        X = transpose(X)

        y = df."Start Distance"
        y = df."Velocity"
    end


    #println(X)
    #println(y)

    options = SymbolicRegression.Options(
        binary_operators=(+, -, /, *),
        #unary_operators=(cos, exp, inv, sin, sqrt),
        unary_operators=(exp, inv),
        #unary_operators=(sqrt, inv),
        npopulations=30,
    )

    hallOfFame = EquationSearch(X, y, niterations=5, options=options, numprocs=4)

    dominating = calculateParetoFrontier(X, y, hallOfFame, options)

    eqn = node_to_symbolic(dominating[end].tree, options)
    #println(simplify(eqn*5 + 3))

    println("Complexity\tMSE\tEquation")

    for member in dominating
        size = countNodes(member.tree)
        score = member.score
        string = stringTree(member.tree, options)

        println("$(size)\t$(score)\t$(string)")
    end

    if mode == BASKETBALL
        println("x1: x-Position Ball, x2: y-Position Ball, x3: Radius, x4: Density, x5: x-Position Basket, x6: y-Position Basket, x7: Radius Basket")
    end
    #df = DataFrame(CSV.File("savedagents/results.csv"))
end

symbolic_regression()