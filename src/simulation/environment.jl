#= 
This file defines the environment type. It basically acts as a 
container for the agents (uavs) and dynamics. 
=#

#=
Environment in which the uavs are making decisions.
=#
type Environment
    max_steps::Int64
    use_noise::Bool
    uavs::Vector{UAV}
    num_uavs::Int64
    gauss_sigma::Float64
    gauss_mu::Float64
    dynamics::Dynamics
    start_state_generator::Function
    min_start_range_ratio::Float64
    max_start_range_ratio::Float64
    range_std::Float64
    initial_heading_offset::Float64
    initial_min_separation::Float64
    delayed_pilot_response::Bool
    rng::MersenneTwister
    state_generator_rng::MersenneTwister
    function Environment(max_steps::Int64, use_noise::Bool, uavs::Vector{UAV}, 
                        gauss_sigma::Float64, gauss_mu::Float64, dynamics::Dynamics,
                        start_state_generator::Function, min_start_range_ratio::Float64, 
                        max_start_range_ratio::Float64, range_std::Float64,
                        initial_heading_offset::Float64, initial_min_separation::Float64, 
                        delayed_pilot_response::Bool, rng::MersenneTwister, 
                        state_generator_rng::MersenneTwister)
        return new(max_steps, use_noise, uavs, length(uavs), gauss_sigma, gauss_mu,
                    dynamics, start_state_generator, min_start_range_ratio, 
                    max_start_range_ratio, range_std, initial_heading_offset, 
                    initial_min_separation, delayed_pilot_response, rng, 
                    state_generator_rng)
    end
end

#=
Description:
Gets actions for each uav based upon the state

Parameters:
- environment: environment in which uavs are acting
- state: current state of the system
- encounter: encounter object tracking information about the system

Return Value:
- uav_actions: Actions type containing both 
    vector of float64 actions indiciating desired bank angles
    as well as vector of signals to be emitted by the uavs

Side Effects:
- alters the encounter object with tracking information
=#
function get_uav_actions!(environment::Environment, state::State, encounter::Encounter)
    # returns UAVActions object for both actions and signals
    uav_actions = UAVActions(state.num_uavs)

    # prepare containers for actions, signals, and encounter tracking info
    num_joint_actions = environment.uavs[1].num_joint_actions
    all_utilities = Array(Float64, (state.num_uavs, num_joint_actions))
    num_states = environment.uavs[1].policy_table.num_states
    belief_states = spzeros(num_states, state.num_uavs)
    
    # delegate action selection to each uav
    for (uidx, uav) in enumerate(environment.uavs)
        action, signals, utilities, belief_state, different_action = get_action(uav, state)

        # if this was the first action advised, then ignore the action
        # because we assume that the pilot takes 5 seconds to respond initially
        # also set the responding state to be responding now so that the pilot
        # responds in the future
        # note that we don't remove signals because those are sent automatically
        # by the system and the pilot response model doesn't impact them
        if (action != COC_ACTION || different_action) && state.responding_states[uidx] == 0
            action = COC_ACTION
            state.responding_states[uidx] = 1
        end

        # set action info
        uav_actions.actions[uidx] = action

        # only take this uavs signals if there are no existing signals
        # this enforces ownship precedence over intruder signalling
        if uav_actions.signals == [:no_signal, :no_signal]
            uav_actions.signals = signals
        end

        # set tracking info
        all_utilities[uidx, :] = utilities
        belief_states[:, uidx] = belief_state
    end

    # update encounter to include actions and other info
    # do so before adding noise so actions denote intended advisories
    update!(encounter, uav_actions, all_utilities, belief_states)

    # add noise to the actions is noisy environment
    if environment.use_noise
        add_noise!(environment, uav_actions.actions)
    end

    # return both the actions and coordination signals
    return uav_actions
end

#=
Description:
steps the state of the system forward in time

Parameters:
- environment: environment in which uavs are acting
- state: current state of the system
- actions: the actions and signals of the uavs
- encounter: encounter object tracking information about the system

Return Values:
- next_state: the next state of the system

Side Effects:
- alters encounter object with tracking information
=#
function step!(environment::Environment, state::State, actions::UAVActions, encounter::Encounter)
    # unpack the uav actions into the two components
    actions, signals = actions.actions, actions.signals

    # delegate to dynamics object to step state forward
    next_state, min_horizontal_dist = step(environment.dynamics, state, actions)

    # propagate the responding state because these are only changed in 
    # get_uav_actions!
    for ridx in 1:state.num_uavs
        next_state.responding_states[ridx] = state.responding_states[ridx]
    end

    # propagate forward the signals to modify utilities the next timestep
    next_state.signals = signals

    # update encounter
    update!(encounter, next_state, min_horizontal_dist)
    return next_state
end

#=
Description:
Returns the initial start state of an encounter. 
Since there are multiple ways of starting an encounter, 
delegate to environments start_state_generator

Parameters:
- environment: the environment in which to start the encounter

Return Value:
- start state: the initial state of the enviornment
=#
function start_encounter(environment::Environment)
    # generate and return a start state
    start_state = environment.start_state_generator(environment)

    # set whether the initial state has responding or not responding
    # pilots - this determines whether there is a delayed pilot response 
    if environment.delayed_pilot_response
        start_state.responding_states = zeros(Int64, start_state.num_uavs)
    else
        start_state.responding_states = ones(Int64, start_state.num_uavs)
    end

    return start_state
end

#=
Description: 
Randomly initializes the state of the system

Parameters:
- environment: environment in which uavs are acting

Return Value:
- initial_state: the initial state of the system
=#
function start_head_on_encounter(environment::Environment)
    # generate random distance
    max_range = environment.uavs[1].coordination_table.grid.cutPoints[1][end]
    max_conflict_range = max_range * environment.max_start_range_ratio
    min_range =  max_range * environment.min_start_range_ratio
    rand_range = rand(environment.state_generator_rng, min_range:1:max_conflict_range)

    # we can always assume that the ownship is at (x,y) == (0,0) because
    # what matters in the simulation is the relative distance between 
    # the uavs
    xown = 0.
    yown = 0.

    # generate random heading for first uav
    heading_stepsize = 2 * pi / 100
    rand_heading_own = rand(environment.state_generator_rng, 0:heading_stepsize:(2 * pi))

    # set the state of the ownship
    max_speed_own = environment.uavs[1].coordination_table.grid.cutPoints[4][end]
    min_speed_own =  environment.uavs[1].coordination_table.grid.cutPoints[4][1]
    speed_own = rand(environment.state_generator_rng, min_speed_own:1:max_speed_own)
    xdot_own = speed_own * cos(rand_heading_own)
    ydot_own = speed_own * sin(rand_heading_own)
    sown = UAVState(xown, xdot_own, yown, ydot_own, rand_heading_own, 0., 0., 0.)

    # given heading of ownship, select a position for the intruder
    # that is on the arc rand_range distance from the ownship
    xint = rand_range * cos(rand_heading_own)
    yint = rand_range * sin(rand_heading_own)

    # generate heading of second uav by starting with an angle
    # moving directly at the first uav and adding to it a 
    # value sampled from normal distribution with mean 0 and std 10 degrees
    rand_heading_int = (rand_heading_own + pi) 
    if environment.use_noise
        rand_heading_int += randn(environment.state_generator_rng) * environment.initial_heading_offset
    end
    
    # correct heading to within 0 to 2 pi
    rand_heading_int = ((rand_heading_int % (2 * pi)) + 2 * pi) % (2 * pi)

    # set the state of the intruder
    max_speed_int = environment.uavs[1].coordination_table.grid.cutPoints[5][end]
    min_speed_int =  environment.uavs[1].coordination_table.grid.cutPoints[5][1]
    speed_int = rand(environment.state_generator_rng, min_speed_int:1:max_speed_int)
    xdot_int = speed_int * cos(rand_heading_int)
    ydot_int = speed_int * sin(rand_heading_int)
    sint = UAVState(xint, xdot_int, yint, ydot_int, rand_heading_int, 0., 0., 0.)

    # create and return the initial state
    return State([sown, sint])
end

#=
Description:
Generates a random uav state for a uav heading toward origin

Parameters:
- environment: the environment in whihc to generate the state
- time_to_origin: time to origin of the first uav
- ownship_range: distance from origin of ownship

Return Value:
- uav_state: the random state for the uav
- time_to_origin: time to origin of the first uav
- ownship_range: distance from origin of ownship
=#
function get_random_uav_state_towards_origin(environment::Environment, 
            time_to_origin, ownship_range)
    # generate random distance
    max_range = environment.uavs[1].coordination_table.grid.cutPoints[1][end]
    max_conflict_range = max_range * environment.max_start_range_ratio
    min_conflict_range =  max_range * environment.min_start_range_ratio
    if ownship_range == nothing
        rand_range = rand(environment.state_generator_rng, min_conflict_range:1:max_conflict_range)
    else
        rand_range = ownship_range + randn(environment.state_generator_rng) * environment.range_std
        rand_range = max(min(rand_range, max_conflict_range), min_conflict_range)
    end

    # generate random heading for first uav
    heading_stepsize = 2 * pi / 100
    rand_heading = rand(environment.state_generator_rng, 0:heading_stepsize:(2 * pi))

    # get x and y position for ship by finding a location on 
    # the circle created around the origin by the random range
    # determined by the heading (so it points towards origin)
    angle_from_origin = (rand_heading + pi) 
    angle_from_origin = ((angle_from_origin % (2 * pi)) + 2 * pi) % (2 * pi)
    x = cos(angle_from_origin) * rand_range
    y = sin(angle_from_origin) * rand_range

    # if no time_to_origin then set speed randomly
    # if time_to_origin, then set speed to have matching time_to_origin
    max_speed = environment.uavs[1].coordination_table.grid.cutPoints[4][end]
    min_speed =  environment.uavs[1].coordination_table.grid.cutPoints[4][1]
    if time_to_origin == nothing
        speed = (max_speed + min_speed) / 2 + randn(environment.state_generator_rng)
    else
        speed = rand_range / time_to_origin
        speed = max(min(speed, max_speed), min_speed)
    end

    # set the state
    xdot = speed * cos(rand_heading)
    ydot = speed * sin(rand_heading)
    uav_state = UAVState(x, xdot, y, ydot, rand_heading, 0., 0., 0.)

    # also calculate the time-to-origin of this plane if not passed in
    if time_to_origin == nothing
        time_to_origin = rand_range / speed
    end
    return uav_state, time_to_origin, rand_range
end

#=
Description: 
Randomly initializes the state of the system.
Loops to avoid initializing the uavs too close to one another.
This could be done analytically, but this is simpler. 

Parameters:
- environment: environment in which uavs are acting

Return Value:
- initial_state: the initial state of the system
=#
function start_towards_origin_encounter(environment::Environment)
    # create initial states for ownship and intruder
    sown, time_to_origin, rand_range = get_random_uav_state_towards_origin(environment, nothing, nothing)
    sint, _, _ = get_random_uav_state_towards_origin(environment, time_to_origin, rand_range)
    state = State([sown, sint])

    # regenerate intruder until it is outside of the min separation distance
    while get_horizontal_distance(state) < environment.initial_min_separation
        sint, _, _ = get_random_uav_state_towards_origin(environment, time_to_origin, rand_range)
        state = State([sown, sint])
    end

    # create and return the initial state
    return state
end

#=
Description:
Adds noise to uav actions

Parameters:
- environment: the environment; this has the noise mu and sigma
- actions: actions from the uavs

Side Effects:
- adds gaussain noise to actions 
=#
function add_noise!(environment::Environment, actions::Vector{Float64})
    for i = 1:length(actions)
        if actions[i] != COC_ACTION
            actions[i] += environment.gauss_mu + environment.gauss_sigma * randn(environment.rng)
        end
    end
end

