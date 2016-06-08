#=
This file contains types that wrap qvalue and coordination tables
and related functions.
=#

using GridInterpolations

push!(LOAD_PATH, ".")
include("state.jl")

#=
PolicyTable type contains all information needed to use a Qvalue table.
=#
type PolicyTable
    q_table::Matrix{Float64}
    grid::RectangleGrid 
    num_states::Int64
    function PolicyTable(q_table::Matrix{Float64}, grid::RectangleGrid, num_states::Int64)
        return new(q_table, grid, num_states)
    end
end

#=
Description:
Use the policy table and belief state to calcualte the utilities of the possible joint actions

Parameters:
- policy_table: the policy table giving Q values
- belief: a sparse matrix of probabilities over states

Return Value:
- utilities: the utilities of the joint actions
    note that these utilities correspond to the joint actions
    based on their order.
=#
function get_utilities(policy_table::PolicyTable, belief_state::SparseMatrixCSC{Float64, Int64})
    # check to make sure that the indices of the belief state correspond
    # with rows of the q_table. If the last row index of the belief state 
    # is in the q_table, then all belief indices must be in the table. 
    if belief_state.rowval[end] > length(policy_table.q_table)
        q_table_size = size(policy_table.q_table)
        msg = "policy_table.q_table of size: $(q_table_size) at index: $(belief_state.rowval[end])"
        throw(BoundsError(msg))
    end

    # create container for the actions
    num_actions = size(policy_table.q_table, 2)
    action_utilities = zeros(num_actions)

    if belief_state[end, 1] != 1
        # iterate over nonzero rows of belief state, multiplying the belief probability
        # that we are in that state by all the values of the q_table
        # summing over these values for the entire table gives the expected value
        # of each action, which we call here the action utilities
        for ridx in belief_state.rowval
            action_utilities += reshape(belief_state[ridx] * policy_table.q_table[ridx, :], num_actions)
        end
    end
    return action_utilities
end

#=
CoordinationTable type contains all information needed to use a coordination table
=#
type CoordinationTable
    c_table::Array{Float64}
    grid::RectangleGrid
    num_states::Int64
    function CoordinationTable(c_table::Array{Float64}, grid::RectangleGrid, num_states::Int64)
        return new(c_table, grid, num_states)
    end
end

#=
Description:
Builds a coordination table using a q table and joint actions matrix. 
Does this by finding the optimal joint action for each row, and 
checking if the planes are turning in the same direction or 
opposite directions. 

Parameters:
- q_table: table of Q values shape (num_states, num_joint_actions)
- joint_actions: matrix where each row corresponds to a uav and each 
    column to a possible joint action

Return Value:
- returns the coordination table
=#
function make_coordiantion_table(q_table::Matrix{Float64}, joint_actions::Matrix{Float64})
    # allocate coordination table
    c_table = zeros(size(q_table, 1), 1)

    # go through each row setting the value
    for ridx in 1:size(q_table, 1)
        utilities = q_table[ridx, :]

        # make it so that we do not select joint actions 
        # with straight or COC as one of the options
        for aidx in 1:size(joint_actions, 2)
            if  joint_actions[1, aidx] == 0. || joint_actions[2, aidx] == 0. || 
                joint_actions[1, aidx] == COC_ACTION || joint_actions[2, aidx] == COC_ACTION
                utilities[aidx] = -Inf
            end
        end

        aidx = indmax(utilities)
        actions = joint_actions[:, aidx]

        # if the signs of the two actions match then 
        # signal 1 for requiring same sense actions
        # note that we removed COC and straight above so 
        # does not come up here therefore case SAME includes:
        # right + right 
        # left + left
        if sign(actions[1]) == sign(actions[2])
            c_table_value = 1

        # otherwise, signal to only do opposite sense
        # therefore case DIFF includes:
        # right + left
        # left + right
        else
            c_table_value = -1
        end

        c_table[ridx] = c_table_value
    end

    return c_table
end


#=
Description:
Determines which sense should be used from the coordination table

Parameters:
- coordination_table: the coordination table to derive the sense from
- belief_state: belief state over the coordination table

Return Value:
- sense: symbol denoting sense
=#
function get_coordination_sense(coordination_table::CoordinationTable, 
                        belief_state::SparseMatrixCSC{Float64, Int64})
    # decide which sense has the most weight on it in the belief state
    sense_same_weight = 0
    sense_diff_weight = 0
    sense_none_weight = 0
    for ridx in belief_state.rowval

        # index into coordination table to see 
        # which sense this belief state corresponds to
        cur_sense = coordination_table.c_table[ridx, 1]

        # get the weight on this state
        belief_weight = belief_state[ridx]

        # accumulate it for the corresponding sense
        if cur_sense == -1 
            sense_diff_weight += belief_weight
        elseif cur_sense == 1
            sense_same_weight += belief_weight
        else
            sense_none_weight += belief_weight
        end
    end

    # and choose whichever sense has highest weight
    if sense_same_weight > sense_diff_weight && sense_same_weight > sense_none_weight
        sense = :same_sense
    elseif sense_diff_weight > sense_none_weight
        sense = :different_sense
    else
        sense = :neither_sense
    end

    return sense
end

#=
Description:
Given two actions, return their sense

Parameters:
- action_1: first action
- action_2: second action

Return Value:
- sense: symbol denoting the sense of the actions
=#
function get_action_sense(action_1::Float64, action_2::Float64)
    # joint actions involving coc and straight are neither sense
    # because no matter what, they are always considered when 
    # deciding which action to take and signal to sen
    if action_1 == COC_ACTION || action_2 == COC_ACTION || 
            action_1 == 0.0 || action_2 == 0.0 
        sense = :neither_sense

    # same sense: right + right or left + left
    elseif sign(action_1) == sign(action_2) 
        sense = :same_sense

    # left + right or right + left
    else
        sense = :different_sense
    end
    return sense
end

