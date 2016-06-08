#=
This file defines the uav type, which uses different strategies to 
make decisions based on the current state of the system.
=#

# type to hold the actions and signals emitted from uavs
type UAVActions
    actions::Vector{Float64}
    # signals sent by the uavs to coordinate next actions
    signals::Vector{Symbol}
    function UAVActions(num_uavs::Int64)
        actions = Array(Float64, num_uavs)
        signals = [:no_signal, :no_signal]
        return new(actions, signals)
    end
end

# type for uav 
type UAV
    # id gives index into tables for this uav 
    uav_id::Int64
    # actions
    actions::Vector{Float64}
    num_actions::Int64
    joint_actions::Matrix{Float64}
    num_joint_actions::Int64
    # tables
    policy_table::PolicyTable
    coordination_table::CoordinationTable
    # strategy
    strategy::Function
    action_selection_method::ASCIIString
    function UAV(uav_id::Int64, actions::Vector{Float64}, joint_actions::Matrix{Float64},
            policy_table::PolicyTable, coordination_table::CoordinationTable, 
            strategy::Function, action_selection_method::ASCIIString)
        return new(uav_id, actions, length(actions), joint_actions, size(joint_actions, 2),
                policy_table, coordination_table, strategy, action_selection_method)
    end
end

#=
Description:
Delegates to the strategy of the uav to decide which action to take

Parameters:
- uav: the uav deciding which action to take
- state: state object which contains all variable information
    needed by the uav to decide which action to take

Return Value:
- action: float value that is the action of this uav
- signal: float value indiciating coordination action
- utilities: the utilities used to derive the actions
- belief_state: the belief_state used to index the policy table
- different_action: whether or not the coordination table prevented
    the taking of a different action from the greedy case and 
    the coordinationt table action was coc. This is used to 
    wake the pilot.
=#
function get_action(uav::UAV, state::State)
    
    # compute belief state 
    belief_state = get_belief_state(uav, state)

    # get the baseline utilities from the policy table
    utilities = get_utilities(uav.policy_table, belief_state)

    # delegate to the strategy to decide which action to take
    # as well as whether to signal the other plane
    action, signals, different_action = uav.strategy(uav, state, utilities)

    return action, signals, utilities, belief_state, different_action
end

#=
Description:
Converts the state to polar format and check that it is in bounds of the policy

Parameters:
- uav: the uav 
- state: state object to convert to polar

Return Value:
- polar_state: the state converted to polar coordinates, which consists of 
    [xr, yr, pr, vown, vint, resp, resp]
=#
function get_polar_state(uav::UAV, state::State)
    # delegate to state for conversion
    polar_state = to_polar(state)

    # get bounds of grid
    max_r = maximum(uav.policy_table.grid.cutPoints[1])
    min_t = minimum(uav.policy_table.grid.cutPoints[2])
    max_t = maximum(uav.policy_table.grid.cutPoints[2])

    # check that the new polar state is inside of the bounds
    # and if not return the terminal state
    r = polar_state[1]
    theta = polar_state[2]
    if r > max_r || theta < min_t || theta > max_t
        polar_state = TERMINAL_POLAR_STATE
    end

    return polar_state
end

#=
Description:
Populates a belief state of the uav

Parameters:
- uav: the uav, the belief state of which is being calculated
- state: the state converted to belief state

Return Value:
- belief_state: probability distribution over possible states
=#
function get_belief_state(uav::UAV, state::State)
    # initialize belief as sparse vector over all possible states
    belief_state = spzeros(uav.policy_table.num_states, 1)

    # convert state to polar
    polar_state = get_polar_state(uav, state)

    # populate belief_state
    if polar_state == TERMINAL_POLAR_STATE
        belief_state[end] = 1.0
    else        
        indices, weights = interpolants(uav.policy_table.grid, polar_state)
        belief_state[indices] = weights
    end

    return belief_state
end

#=
Description:
Returns the id value of the other uav 

Parameters:
- uav: the current uav

Return Value:
- id: int value of the other uavs id
=#
function get_other_uav_id(uav::UAV)
    if uav.uav_id == 1
        return 2
    else
        return 1
    end
end


