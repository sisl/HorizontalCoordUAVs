#=
This file defines the encounter type, which is used to
track information about an encounter in a simulation.
All information that should be tracked, but that is not
necessary for the uavs to make decisions or for the dynamics
to step the system forward should be stored in an encounter.
=#

#=
This type is used to track the trajectory of a uav.
=#
type Path
    x_coordinates::Vector{Float64}
    y_coordinates::Vector{Float64}
end

#=
Encounter type tracks information about an encounter.
=#
type Encounter
    paths::Vector{Path}
    actions::Matrix{Float64}
    signals::Array{Symbol}
    min_horizontal_dist::Float64
    utilities::Array{Array{Float64}}
    belief_states::Array{SparseMatrixCSC}
    states::Array{State}
    min_horizontal_dists::Array{Float64}
    nmac::Bool
    function Encounter()
        paths = [Path([], []), Path([], [])]
        actions = Array(Float64, 2, 0)
        signals = [:no_signal, :no_signal]
        return new(paths, actions, signals, Inf, [], [], [], [], false)
    end

end

#=
Description:
Updates the encounter object to include the actions

Parameters:
- encounter: Encounter type to add actions to
- actions: actions to add 

Side Effects:
- updates encounter to include the actions
=#
function update!(encounter::Encounter, actions::Vector{Float64})
    encounter.actions = hcat(encounter.actions, actions)
end

#=
Description:
Updates the encounter object to include the actions

Parameters:
- encounter: Encounter type to add actions to
- actions: actions to add 
- utilities: the utilities for both uavs
- belief_state: the belief_state for both uavs

Side Effects:
- updates encounter to include the actions, beliefs, utilities
=#
function update!(encounter::Encounter, uav_actions::UAVActions, 
            utilities::Array{Float64}, belief_state::SparseMatrixCSC)
    encounter.actions = hcat(encounter.actions, uav_actions.actions)
    encounter.signals = hcat(encounter.signals, uav_actions.signals)
    push!(encounter.utilities, utilities)
    push!(encounter.belief_states, belief_state)
end

#=
Description:
Updates the encounter object to include the uav x and y coordinates
from the state.

Parameters:
- encounter: Encounter type to add actions to
- state: state containing x and y coordinates to add
- min_horizontal_dist: minimum horz dist to add to list

Side Effects:
- updates encounter to include the x and y coordinates and horz dist
- also to include state, min_horizontal_dist
=#
function update!(encounter::Encounter, state::State, min_horizontal_dist::Float64 = -1.)
    # update trajectory information
    push!(encounter.states, state)
    for (uidx, uav_state) in enumerate(state.uav_states)
        push!(encounter.paths[uidx].x_coordinates, uav_state.x)
        push!(encounter.paths[uidx].y_coordinates, uav_state.y)
    end 

    # update horizontal distance array
    if min_horizontal_dist == -1.
        min_horizontal_dist = get_horizontal_distance(state)
    end
    push!(encounter.min_horizontal_dists, min_horizontal_dist)
end
