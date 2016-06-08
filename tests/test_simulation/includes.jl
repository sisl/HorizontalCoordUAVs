
push!(LOAD_PATH, "../../src/simulation/")

include("../../src/simulation/tables.jl")
include("../../src/simulation/state.jl")
include("../../src/simulation/dynamics.jl")
include("../../src/simulation/uav.jl")
include("../../src/simulation/encounter.jl")
include("../../src/simulation/strategies.jl")
include("../../src/simulation/environment.jl")
include("../../src/simulation/build_simulation.jl")
include("../../src/simulation/analyze_encounters.jl")

# some functions and constants common to multiple test files
const SHOW_PLOTS = true

function get_real_uav()
    opts = SimulationOptions()
    uavs = build_uavs(opts)
    return uavs[1]
end