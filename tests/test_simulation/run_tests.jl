
push!(LOAD_PATH, ".")

include("test_dynamics.jl")
include("test_encounter.jl")
include("test_environment.jl")
include("test_state.jl")
include("test_strategies.jl")
include("test_tables.jl")
include("test_uav.jl")

println("All tests pass!")