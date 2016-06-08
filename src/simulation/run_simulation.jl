#=
The main script that simulates encounters.
=#

push!(LOAD_PATH, ".")

include("tables.jl")
include("state.jl")
include("dynamics.jl")
include("uav.jl")
include("encounter.jl")
include("strategies.jl")
include("environment.jl")
include("build_simulation.jl")
include("analyze_encounters.jl")

#=
Description:
This method simulates a single encounter. Abstractly, this
entails initializing an encounter, simulating
each uav in the environment taking actions, and then 
simulating the trajectories resulting from these actions.

Parameters:
- environment: the environment in which to simulate the encounter

Return Value:
- encounter: a single encounter object

=#
function simulate_encounter(environment::Environment)
    # select a random start state
    state = start_encounter(environment)

    # create an encounter to track information
    encounter = Encounter()
    update!(encounter, state)

    # simulate the actual encounter
    for idx in 1:environment.max_steps
        actions = get_uav_actions!(environment, state, encounter)
        state = step!(environment, state, actions, encounter)
    end

    return encounter
end


#=
Description:
This method simulates a variable number of encounters using the 
environment provided to it.

Parameters:
- environment: the environment in which to simulate encounters
- num_encounters: the number of encounters to simulate

Return Value:
- encounters: a list of encounter objects which contain all the 
                information needed to visualize each encounter
=#
function simulate_encounters(environment::Environment, num_encounters::Int64)
    
    # simulate num_encounters, collecting encounter information
    encounters = Array(Encounter, num_encounters)
    for idx in 1:num_encounters
        encounters[idx] = simulate_encounter(environment)
    end
    return encounters
end 

#=
Description:
Main method that performs setup and then calls simulate_encounters
to actually simulate encounters.
=#
function run_simulation()
    # get the environment to simulate encounters in
    environment = build_environment()

    # simulate the encounters
    num_encounters = 10
    encounters = simulate_encounters(environment, num_encounters)

    # analyze the encounters
    analyze(encounters)
end

# run it
@time run_simulation()

