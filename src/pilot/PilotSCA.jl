# load dependencies
push!(LOAD_PATH, "../dvi")
push!(LOAD_PATH, ".")

# replace with the desired number of processes
numProcs = 2
addprocs(numProcs)

using DiscreteValueIteration, JLD, PilotSCAs

mdp = SCA()
solver = ParallelSolver(
    numProcs,
    maxIterations = 200,
    tolerance = 1e-4,
    gaussSiedel = false,  # true
    includeV = true,
    includeQ = true,
    includeA = true)

println("\nStarting parallel solver...")
policy = solve(solver, mdp, verbose = true)
println("\nParallel solution generated!")

# save solution
function sharray2array(sharray::SharedArray{Float64, 2})
    result = zeros(sharray.dims)
    for i = 1:sharray.dims[1]
        for j = 1:sharray.dims[2]
            result[i, j] = sharray[i, j]
        end # for j
    end # for i
    return result
end # function sharray2array

function sharray2array(array::Array{Float64, 2})
    return array
end # function sharray2array

solQ = sharray2array(policy.Q')
save("../../data/qvalue_tables/pilot-alpha.jld", "solQ", solQ)
println("Parallel solution saved...exiting PilotSCA.jl script.")
