#=
This file contains the different strategies (i.e., algorithms) that
the uavs can use.
=#

#=
Description:
A strategy that uses a coordination table to decide which action to take.

Parameters:
- uav: the uav deciding which action it should take
- state: the current state of the system, which contains 
    all variable information needed to optimally (given the 
    constraints of the environment) decide which action to take 
- utilities: the utilities of the different joint actions

Return Value:
- action: the action this uav should take, which is a float
- signals: the signals to send for both the current uav and other uav
- different_action: whether or not the coord table prevent a non coc action
=#
function coordinated_strategy(uav::UAV, state::State, utilities::Vector{Float64})
    # track whether the coordination table took a different action with this flag
    different_action = false

    # check if all the utilities are the same, in which case, take no action
    # this happens when we are in the "terminal" state of the table
    # i.e., one outside the bounds of the table
    if length(unique(utilities)) == 1
        action = COC_ACTION
        signals = [:no_signal, :no_signal]
    else
        # modify utilities based on signal 
        modified_utilities = modify_utilities(uav, state, utilities)

        # get sense: :same_sense or :different_sense
        sense = get_coordination_sense(uav, state)

        # get action and signal
        prev_signals = state.signals
        action, signals = select_action_and_signals(uav, modified_utilities, sense, prev_signals)

        # check what the greedy action would be
        greedy_action_idx = indmax(utilities)
        greedy_action = uav.joint_actions[uav.uav_id, greedy_action_idx]

        if action == COC_ACTION && greedy_action != COC_ACTION
            different_action = true
        end

    end

    # return action and signals for both planes for the next timestep
    return action, signals, different_action
end

#=
Description:
Based on the coordination signals of the uavs, modifies utilities 
that do not follow coordination signals to be -Inf, effectively
preventing the uav from taking this action. 

Parameters:
- uav: the uav choosing an action
- state: state of the system
- utilities: the estimated q_values for the different joint_actions

Return Value:
- returns modified utility values that align with the coordination scheme
=#
function modify_utilities(uav::UAV, state::State, utilities::Vector{Float64})
    # if no signals emitted do not alter utilities
    # if the signal emitted was by this uav, then it should 
    # not alter utilities because only one signal can be 
    # active at a time
    if all(state.signals .== :no_signal)
        return utilities
    end

    # reaching this point means a signal was sent last 
    # timestep that needs to be followed this timestep
    # signal is either :dont_turn_left or :dont_turn_right
    signal = state.signals[uav.uav_id]

    # go through each joint action adjusting value based on signal
    for aidx in 1:size(uav.joint_actions, 2)

        # determine whether the action for the current 
        # uav in this joint action is right, left, or neither
        cur_uav_action = uav.joint_actions[uav.uav_id, aidx]

        # uav.actions[1 or 2] are right turns
        if cur_uav_action == uav.actions[1] || cur_uav_action == uav.actions[2]
            turn = :right

        # uav.actions[4 or 5] are left turns
        elseif cur_uav_action == uav.actions[4] || cur_uav_action == uav.actions[5]
            turn = :left

        # straight or coc
        else
            turn = :neither
        end

        # if the current uavs turn in this joint action 
        # is in the direction it has been signalled not 
        # to turn, then set that actions utility to 0
        if (turn == :right && signal == :dont_turn_right) ||
                (turn == :left && signal == :dont_turn_left)
            utilities[aidx] = -Inf
        end
    end

    return utilities
end

#=
Description:
Determines what sense to use in deciding action and signal

Parameters:
- uav: uav deciding what action and signal to execute
- state: state of the system

Return Value:
- sense: one of {-1 == different, +1 == same}
=#
function get_coordination_sense(uav::UAV, state::State)

    # otherwise use belief_state to decide which sense to use
    belief_state = get_belief_state(uav, state)

    # retrieve best sense from table
    sense = get_coordination_sense(uav.coordination_table, belief_state)

    return sense
end

#=
Description:
Choose an action for this uav and signals to send to the next timestep.

Parameters:
- uav: uav deciding which action to select
- utilities: the modified utilities for different actions
- sense: sense from coordination table
- prev_signals: the signals from the preceding timestep

Return Value:
- action: the action to take
- signals: the signals to send to the uavs at the next timestep
=#
function select_action_and_signals(uav::UAV, utilities::Vector{Float64}, 
            sense::Symbol, prev_signals::Array{Symbol})

    # alias this value for easier reference
    method = uav.action_selection_method

    # based on the method, select the way to accumulate across intruder actions
    # as well as the initial values of the dictionaries
    if method == "best"
        accumulator = max
    elseif method == "worst"
        accumulator = min
    elseif method == "average"
        accumulator = +
    else
        throw(ArgumentError("invalid action selection method $(method)"))
    end

    # create two dicts mapping actions to values
    # and initially populate with +infinity, -infinity, or 0
    # one for ownship going left and another for ownship going right
    ownship_left_action_values = Dict{Float64, Float64}()
    ownship_right_action_values = Dict{Float64, Float64}()

    # iterate over utilities populating dicts with values
    # enforce the coordination component here:
    # if the joint action is an invalid sense then skip it
    # because we can be sure that the other plane can be 
    # prevented from taking that action
    for aidx in 1:size(uav.joint_actions, 2)
        own_action = uav.joint_actions[uav.uav_id, aidx]
        int_action = uav.joint_actions[get_other_uav_id(uav), aidx]
        action_sense = get_action_sense(own_action, int_action)

        # if the sense on this joint action is incorrect, then skip it
        if (sense != :neither_sense && action_sense != :neither_sense 
                && action_sense != sense)
            continue
        end

        # update the min value corresponding to this ownship action
        # if this action is _left_ or straight (including coc), then 
        # that means it should go in the _left_ dict of the ownship
        if sign(own_action) == 1 || sign(own_action) == 0 || own_action == COC_ACTION

            if own_action in keys(ownship_left_action_values)
                ownship_left_action_values[own_action] = 
                    accumulator(ownship_left_action_values[own_action], utilities[aidx])
            else
                ownship_left_action_values[own_action] = utilities[aidx]
            end
        end

        # update the min value corresponding to this ownship action
        # if this action is _right_ or straight (including coc), then 
        # that means it should go in the _right_ dict of the ownship
        if sign(own_action) == -1 || sign(own_action) == 0 || own_action == COC_ACTION
            if own_action in keys(ownship_right_action_values)
                ownship_right_action_values[own_action] = 
                    accumulator(ownship_right_action_values[own_action], utilities[aidx])
            else
                ownship_right_action_values[own_action] = utilities[aidx]
            end
        end
    end

    # get the best action and the value of the worst action in each dict
    left_action, left_value = get_best_action_value(ownship_left_action_values, method)
    right_action, right_value = get_best_action_value(ownship_right_action_values, method)

    # comment on best case, with best action being straight for both:
    # the left value captures the max across intruder actions for straight
    # and so does the right value
    # so if one is greater than the other then we just take that signal (dlt or dlr)
    # if they are equal in value, then that means the best action for both was 
    # straight + (straight or coc), in which case we either don't send a signal 
    # or just propagate the signal from the previous timestep, which is accounted 
    # for below in a separate if statement

    # left value greater
    if left_value > right_value
        own_action = left_action
        own_signal = :dont_turn_right
        int_signal = sense == :same_sense ? :dont_turn_right : :dont_turn_left
    
    # otherwise go with right action
    else
        own_action = right_action
        own_signal = :dont_turn_left
        int_signal = sense == :same_sense ? :dont_turn_left : :dont_turn_right
    end

    # if ownship action is straight and there was a previous coordination signal
    # then just propagate those signals
    # if there was not a previous signal then this does not come into play
    if own_action == 0.0 && (prev_signals[1] != :no_signal || prev_signals[2] != :no_signal)
        own_signal = prev_signals[1]
        int_signal = prev_signals[2]
    end

    # if coordination table gives neither sense then don't send signals
    if sense == :neither_sense || own_action == COC_ACTION
        own_signal = :no_signal
        int_signal = :no_signal
    end

    signals = [own_signal, int_signal]

    return own_action, signals
end

#=
Description:
Given a dictionary mapping actions to values, returns the best 
action from the dictionary and the minimum value across actions 
in the dictionary.

Parameters:
- action_value_dict: the dictionary to search for the best action

Return Values:
- best_action: best action
- return_value: value to be returned. 
    best_case = value of best action
    worst_case = minimum value across the actions in the dict
    average_case = sum of values across the actions in the dict
=#
function get_best_action_value(action_value_dict::Dict{Float64, Float64}, 
            method::ASCIIString)
    # choose actual action by going through dict and choosing action
    # that has either the largest value
    # in the best case, we just care what the max value is
    # in the worst case, we care what the min value across the dict is
    # in the average case, we care what the expected value across the dict is
    best_action = COC_ACTION
    max_value = -Inf
    min_value = Inf
    avg_value = 0
    for action in keys(action_value_dict)
        cur_value = action_value_dict[action]

        # best case
        if cur_value > max_value
            best_action = action
            max_value = cur_value
        end

        # worst case
        if cur_value < min_value
            min_value = cur_value
        end

        # average case
        avg_value += cur_value
    end

    # choose which value to return
    return_value = max_value

    return best_action, return_value
end

#=
Description:
A strategy that determines the optimal joint action 
independently from the other uavs.

Parameters:
- uav: the uav deciding which action it should take
- state: the current state of the system
- utilities: the utilities of the different joint actions

Return Value:
- action: the action this uav should take, which is a float
- signals: the signals to send for both the current uav and other uav
- different_action: whether or not the coord table prevent a non coc action
=#
function greedy_strategy(uav::UAV, state::State, utilities::Vector{Float64})
    # track whether the actions changed due to coordination
    different_action = false

    # check if all the utilities are the same, in which case, take no action
    if length(unique(utilities)) == 1
        action = COC_ACTION
    else
        # always modify utilities in case this is the intruder
        # but will not impact anything if singals are both no_signal
        modified_utilities = modify_utilities(uav, state, utilities)
        best_action_idx = indmax(modified_utilities)
        action = uav.joint_actions[uav.uav_id, best_action_idx]

        # check what the greedy action would be
        greedy_action_idx = indmax(utilities)
        greedy_action = uav.joint_actions[uav.uav_id, greedy_action_idx]

        if action == COC_ACTION && greedy_action != COC_ACTION
            different_action = true
        end
    end
    
    # since this strategy does not use coordinated actions,
    # return no signals
    # return different_action to have same interface as 
    # the coordinated strategy, as well as to wake intruder pilot
    # in the case that the ownship is running coordinated strategy
    # and intruder runs compliant greedy strategy
    return action, [:no_signal, :no_signal], different_action
end

