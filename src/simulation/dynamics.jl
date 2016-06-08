#=
This file defines the dynamics type. It is used to 
step the state of the system forward in time.
=#

using ODE

#=
Dynamics type contains constant values used for progressing the 
state of the system forward in time. 
=#
type Dynamics
    dt::Int64
    dt_step::Float64
    pid_omega::Float64
    pid_b_std::Float64
    pid_b_std_coc::Float64
    use_noise::Bool
    rng::MersenneTwister
    function Dynamics(dt::Int64, dt_step::Float64, pid_omega::Float64, 
                pid_b_std::Float64, pid_b_std_coc::Float64, use_noise::Bool, 
                rng::MersenneTwister)
        return new(dt, dt_step, pid_omega, pid_b_std, pid_b_std_coc, use_noise, rng)
    end
end

#=
Description:
This function steps the state of the system forward using the actions of the uavs

Parameters:
- dynamics: Dynamics type containing necessary fields
- state: State type containing all variable information necessary for progressing
        the system except for the agent actions
- actions: The actions of the uavs in the system

Return Values:
- next_state: the next state of the system after stepping forward in time
- min_horizontal_dist: minimum horizontal distance experienced by the uavs
=#
function step(dynamics::Dynamics, state::State, actions::Vector{Float64})
    @assert state.num_uavs == length(actions) "Each uav must have one action. \
        state: $state\tactions: $actions"

    # generate pid controllers for the different uavs using their actions
    policies = Array(Function, state.num_uavs)
    for idx in 1:state.num_uavs
        action = actions[idx]

        # correct COC to 0 degrees nominal turn
        coc_flag = false
        if action == COC_ACTION
            coc_flag = true
            action = 0.0
        end

        # add noise to bank angle commanded
        if dynamics.use_noise
            if coc_flag
                bank_angle = action + dynamics.pid_b_std_coc * randn(dynamics.rng)
            else
                bank_angle = action + dynamics.pid_b_std * randn(dynamics.rng)
            end
        else
            bank_angle = action
        end
        
        policies[idx] = build_pid_policy(dynamics, bank_angle)
    end 

    # step forward each uav independently
    # collect two things:
    # first, the final result of the dynamics applied to each uav
    # second, all the intermediate states also
    # use the intermediate states to determine the min_horizontal_dist
    next_uav_states = Array(UAVState, state.num_uavs)
    intermediate_uav_states = Array(Array{UAVState}, state.num_uavs)

    for idx in 1:state.num_uavs
        next_states = step(state.uav_states[idx], policies[idx], dynamics.dt, dynamics.dt_step)
        next_uav_states[idx] = next_states[end]
        intermediate_uav_states[idx] = next_states
    end

    # get the min_horizontal_dist to return
    min_horizontal_dist = get_min_horizontal_distance(intermediate_uav_states)

    # create and return a new State object using the list of UAVStates
    return State(next_uav_states), min_horizontal_dist
end

#=
Description:
This function steps forward the state of a single uav using a 
policy (i.e., pid controller)

Parameters:
- uav_state: a UAVState object for the uav to be stepped forward
- policy: the pid controller for this uav
- dt: time steps to step forward
- dt_step: number of dynamics updates per timestep

Return Value:
- next_states: UAVState objects containing the intermediate states of the uav
=#
function step(uav_state::UAVState, policy::Function, dt::Int64, dt_step::Float64)
    # convert to vector form, call substep, and return
    uav_state_vector = convert(Vector{Float64}, uav_state)
    next_states = substep(uav_state_vector, policy, dt, dt_step)
    return next_states
end

#=
Description:
Steps forward the state by the smallest unit of time

Parameters:
- uav_state_vector: the state of the uav as a vector instead of an object
- policy: the pid controller for this uav
- dt: time steps to step forward
- dt_step: number of dynamics updates per timestep

Return Values:
- next states: the next states in UAVState format
=#
function substep(uav_state_vector::Vector{Float64}, policy::Function, dt::Int64, dt_step::Float64)
    # use the actual speed of the uav rather than the speed constant
    speed = get_speed(uav_state_vector)
    G = 9.8

    # dynamics equation to solve
    function dxdt(t::Float64, state::Vector{Float64})
        dx = zeros(length(state))
        dx[1] = speed * cos(state[5])                   # xdot
        dx[3] = speed * sin(state[5])                   # ydot
        dx[5] = G * tan(state[7]) / speed               # psidot
        dx[7] = state[8]                                # phidot
        dx[8] = policy(state)                           # phiddot
        dx[2] = -speed * sin(state[5]) * dx[7]          # xddot
        dx[4] = speed * cos(state[5]) * dx[7]           # yddot
        dx[6] = G / speed * sec(state[8])^2 * dx[8]     # psiddot
        return dx
    end

    # create a list of the times to step through
    times = 0:dt_step:dt

    # step through them solving for next state each timestep
    # initial state is the incoming state vector
    state = uav_state_vector

    # collect the states as row vectors in an array
    states = Array(UAVState, length(times) - 1)
    for tidx = 1:length(times) - 1
        tspan = [times[tidx], times[tidx + 1]]
        _, xout = ODE.ode45(dxdt, state, tspan) 

        # collect states to return 
        states[tidx] = convert(UAVState, xout[end])

        # set the state for use in the next loop 
        state = xout[end]
    end

    # return all intermediate states for min horizontal computation
    return states
end

#=
Description:
Builds a pid controller given an action and is an effective policy for the uav

Parameters:
- dynamics: the dynamics of the environment in this case provide constant values
- action: the bank angle the uav has decided to take (in radians)

Return Value:
- pid_policy: a function that given a state returns an action
=#
function build_pid_policy(dynamics::Dynamics, action::Float64)
    function pid_policy(uav_state_vector::Vector{Float64})
        policy =  2 * dynamics.pid_omega * -uav_state_vector[8] + 
                    dynamics.pid_omega^2 * (action - uav_state_vector[7])
        return policy
    end
    return pid_policy
end

