using Pkg
using SymbolicRegression
using CSV
using DataFrames

random_densities = true

df = DataFrame(CSV.File("savedagents/results7.csv"))
df = select!(df, Not("Column1")) # cut the first column
#println(df)

if (random_densities)
    X = Matrix(df[:, ["Position 1", "Density 1", "Density 2", "Boxsize 1", "Boxsize 2"]])
else
    X = Matrix(df[:, ["Position 1", "Density 1", "Density 2"]])
end
X = transpose(X)
y = df."Position 2"

#println(X)
#println(y)

options = SymbolicRegression.Options(
    binary_operators=(+, *),
    #unary_operators=(cos, exp, inv, sin),
    unary_operators=(exp, inv),
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
end