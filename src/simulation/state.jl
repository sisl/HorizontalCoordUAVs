#=
This file defines the State type, which maintains the state of the
systems and is provided to the other components in order to allow
then to either step the system forward or make decisions based on 
the state.
=#

# const values
const COC_ACTION = -1.

# this assumes that we do not include 
# responding / not responding
# but this does not influence simulator 
# because it is used as a flag, not for
# indexing into any tables
const TERMINAL_POLAR_STATE = ones(5) * 1e5

#=
State associated with a single uav.
=#
type UAVState
    x::Float64
    xdot::Float64
    y::Float64
    ydot::Float64
    psi::Float64 # heading
    psidot::Float64
    phi::Float64 # bank angle
    phidot::Float64
    function UAVState(x::Float64, xdot::Float64, y::Float64, ydot::Float64, 
                        psi::Float64, psidot::Float64, phi::Float64, phidot::Float64)
        return new(x, xdot, y, ydot, psi, psidot, phi, phidot)
    end
end

# convert a UAVState type to a vector
function convert(::Type{Vector{Float64}}, uav_state::UAVState)
    return [uav_state.(f) for f in fieldnames(uav_state)]
end

# convert a vector to a UAVState type
function convert(::Type{UAVState}, uav_state_vector::Vector{Float64})
    @assert length(uav_state_vector) == length(fieldnames(UAVState)) "vector must have the \
        same number of elements as there are fieldnames in the UAVState type"
    return UAVState(uav_state_vector...)
end

#=
Description:
Return the speed of the uav, calculated from state variables

Parameters:
uav_state_vector: state of uav as a vector

Return Value:
speed: float value for the speed of the uav
=#
function get_speed(uav_state_vector::Vector{Float64})
    return sqrt(uav_state_vector[2]^2 + uav_state_vector[4]^2)
end

#=
Description:
Convenience method for determining if two uav states are equal

Parameters:
- u1: first uav state
- u2: second uav state

Return Values:
- whether or not the two uav states are the same
=#
function Base.(:(==))(u1::UAVState, u2::UAVState)
    return convert(Vector{Float64}, u1) == convert(Vector{Float64}, u2)
end 

#=
The State type contains all the non-constant information needed by the uavs 
to decide what action to take and all the non-constant information needed by 
the dynamics type to step the entire environment forward in time.
=#
type State
    uav_states::Vector{UAVState}
    num_uavs::Int64
    # these are the coordination signals 
    # emitted the previous timestep
    signals::Array{Symbol}
    responding_states::Vector{Int64}
    function State(uav_states::Vector{UAVState})
        num_uavs = length(uav_states)
        signals = [:no_signal, :no_signal]
        responding_states = zeros(Int64, num_uavs)
        return new(uav_states, num_uavs, signals, responding_states)
    end
end

#=
Description:
Convert a state object from cartesian to polar coordinates.
Note that this function does not do bounds checking on the result.
This is performed after calling this function, in the uav 
get_polar_state call.

Parameters:
- state: state object to be converted

Return Value:
- polar_state: the state in polar format
=#
function to_polar(state::State)
    @assert length(state.uav_states) == 2 "invalid number of uav_states in state: $state"

    sown = state.uav_states[1]
    sint = state.uav_states[2]

    dx = sint.x - sown.x
    dy = sint.y - sown.y
    xr = dx * cos(sown.psi) + dy * sin(sown.psi)
    yr = -dx * sin(sown.psi) + dy * cos(sown.psi)

    # heading is relative to ownship 
    # and is caluated by subtracting the 
    # heading of the intruder and normalizing
    # the resulting angle to between
    # 0 and 2pi
    pr = norm_angle(sint.psi - sown.psi)

    vown = sqrt(sown.xdot^2 + sown.ydot^2)
    vint = sqrt(sint.xdot^2 + sint.ydot^2)

    r = sqrt(xr^2 + yr^2)
    # norm theta to be between -pi and pi
    theta = norm_angle(atan2(yr, xr), neg_pi_to_pi = true)

    own_responding = state.responding_states[1]
    int_responding = state.responding_states[2]

    polar_state = [r, theta, pr, vown, vint, own_responding, int_responding]
    return polar_state

end

#=
Description:
Normalize an angle between 0 and 360
If neg_pi_to_pi flag is true, then between -pi and pi

Parameters:
- angle: angle to Normalize
- neg_pi_to_pi: normalize to between -pi and pi

Return Value:
- noramlized angle
=#
function norm_angle(angle::Float64; neg_pi_to_pi::Bool = false)
    new_angle = ((angle % (2 * pi)) + 2 * pi) % (2 * pi)
    if neg_pi_to_pi
        if new_angle > pi
            new_angle -= 2 * pi
        end
    end
    return new_angle
end

#=
Description:
Returns the horizontal distance between the two uavs

Parameters:
- state: current state of the system

Return Values:
- horizontal_distance: the horizontal distance between the two uavs
=#
function get_horizontal_distance(state::State)
    sown = state.uav_states[1]
    sint = state.uav_states[2]
    horizontal_distance = sqrt((sown.x - sint.x)^2 + (sown.y - sint.y)^2)
    return horizontal_distance
end

#=
Description: 
Get the minimum horizontal distance that occurred in a set of 
intermediate states.

Parameters:
- state: the intermediate uav states (shape = (num_uavs, num states))

Return Value:
- min horizontal distance: min horizontal dist in meters, float
=#
function get_min_horizontal_distance(states::Array{Array{UAVState}})
    min_horizontal_dist = Inf
    for (sown, sint) in zip(states[1], states[2])
        horizontal_distance = sqrt((sown.x - sint.x)^2 + (sown.y - sint.y)^2)
        min_horizontal_dist = min(min_horizontal_dist, horizontal_distance)
    end
    return min_horizontal_dist
end

#=
Description:
Convenience method for determining if two states are equal

Parameters:
- s1: first state
- s2: second state

Return Values:
- whether or not the two states are the same
=#
function Base.(:(==))(s1::State, s2::State)
    if s1.num_uavs != s2.num_uavs
        return false
    end
    for (u1, u2) in zip(s1.uav_states, s2.uav_states)
        if u1 != u2 
            return false
        end
    end
    return true
end 



