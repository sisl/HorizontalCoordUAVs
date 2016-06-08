# tests for the State and UAVState types and related functions.

using Base.Test

push!(LOAD_PATH, ".")
include("includes.jl")

# State tests
function test_constructor()
    uav_state = UAVState(collect(1.:8)...)
    s = State([uav_state])
    @test s.num_uavs == 1
end

function test_to_polar()
    uav_state_own = UAVState(ones(Float64, 8)...)
    uav_state_int = UAVState(ones(Float64, 8)...)
    s = State([uav_state_own; uav_state_int])
    s.responding_states = [1,1]
    cvec = to_polar(s)
    @test cvec == [0,0,0,sqrt(2),sqrt(2),1,1]

    uav_state_1 = UAVState(0., 10., 0., 10., 0., 0., 0., 0.)
    uav_state_2 = UAVState(100., 10., 100., 10., .2, 0., 0., 0.)
    state = State([uav_state_1, uav_state_2])
    state.responding_states = [1,1]
    cvec = to_polar(state)
    r = sqrt(100^2 + 100^2)
    theta = atan2(100, 100)
    heading = norm_angle(.2 - 2 * pi)
    vown = sqrt(200)
    vint = sqrt(200)
    resown = 1
    resint = 1
    polar_state = [r, theta, heading, vown, vint, resown, resint]
    @test cvec == polar_state

end

function test_state_equality()
    uav_state_1 = UAVState(1.,2.,3.,4.,5.,6.,7.,8.)
    uav_state_2 = UAVState(1.,2.,3.,4.,5.,6.,7.,8.)
    uav_state_3 = UAVState(0.,0.,0.,0.,0.,0.,0.,0.)
    state_1 = State([uav_state_1, uav_state_2])
    state_1.responding_states = [1,1]
    state_2 = State([uav_state_1, uav_state_2])
    state_2.responding_states = [1,1]
    @test state_1 == state_2
    state_3 = State([uav_state_1, uav_state_3])
    state_3.responding_states = [1,1]
    @test state_1 != state_3
end

# UAVState tests
function test_convert_to_vector()
    uav_state = UAVState(1.,2.,3.,4.,5.,6.,7.,8.)
    v = convert(Vector{Float64}, uav_state)
    @test v == collect(1.:8)
end

function test_convert_from_vector()
    uav_state = UAVState(collect(1.:8)...)
    @test uav_state.x == 1.
    @test uav_state.xdot == 2.
    @test uav_state.y == 3.
    @test uav_state.ydot == 4.
    @test uav_state.psi == 5.
    @test uav_state.psidot == 6.
    @test uav_state.phi == 7.
    @test uav_state.phidot == 8.
end

function test_get_speed()
    uav_state_vector = collect(1.:8)
    @test get_speed(uav_state_vector) == sqrt(2^2 + 4^2)
end

function test_get_horizontal_distance()
    uav_state_own = UAVState(ones(Float64, 8)...)
    uav_state_int = UAVState(zeros(Float64, 8)...)
    s = State([uav_state_own; uav_state_int])
    d = get_horizontal_distance(s)
    @test d == sqrt(2)
end

function test_uav_state_equality()
    uav_state_1 = UAVState(1.,2.,3.,4.,5.,6.,7.,8.)
    uav_state_2 = UAVState(1.,2.,3.,4.,5.,6.,7.,8.)
    @test uav_state_1 == uav_state_2
    uav_state_3 = UAVState(0.,0.,0.,0.,0.,0.,0.,0.)
    @test uav_state_1 != uav_state_3
end

function main()
    # State tests
    test_constructor()
    # commented tests out of order until decide on polar formulation
    test_to_polar()
    test_state_equality()

    # UAVState tests
    test_convert_to_vector()
    test_convert_from_vector()
    test_get_speed()
    test_get_horizontal_distance()
    test_uav_state_equality()
end

@time main()