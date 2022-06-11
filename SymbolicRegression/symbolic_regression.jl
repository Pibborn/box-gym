using Pkg
using SymbolicRegression
using CSV
using DataFrames

SCALE = 1
BASKETBALL = 2
mode = BASKETBALL

file_name = "resultsBasketbalL"
df = DataFrame(CSV.File("savedagents/extracted_data/$file_name.csv"))
df = select!(df, Not("Column1")) # cut the first column

if (mode == SCALE)
    random_boxsizes = true
    actions = 1
    placed = 2
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
    y = df."Position 3"
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
   X = Matrix(df[:, ["x-Position Ball", "y-Position Ball", "Force vector x", "Radius", "Density", "x-Position Basket", "y-Position Basket", "Radius Basket"]])
   X = transpose(X)

   # old
   # y = df."x-Position Ball"
   # new
   # y = df[:, ["Force vector x", "Force vector y"]]
   y = df."Force vector y"
end


#println(X)
#println(y)

options = SymbolicRegression.Options(
    binary_operators=(+, *),
    unary_operators=(cos, exp, inv, sin),
    #unary_operators=(exp, inv),
    npopulations=20,
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

#df = DataFrame(CSV.File("savedagents/results.csv"))
"""
X = randn(Float32, 5, 100)
y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2

options = SymbolicRegression.Options(
    binary_operators=(+, *, /, -),
    unary_operators=(cos, exp),
    npopulations=20
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
"""
    #println("$(size)\t$(score)\t$(string)")
