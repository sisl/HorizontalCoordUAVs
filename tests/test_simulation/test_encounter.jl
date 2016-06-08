# tests for the encounter type and related functions.

using Base.Test

push!(LOAD_PATH, ".")
include("includes.jl")

# encounter tests
function test_constructor()
    e = Encounter()
end

# update! tests
function test_update_actions()
    e = Encounter()
    uav_actions = UAVActions(2)
    uav_actions.actions = deg2rad([-10., -10.])
    utilities = [[[1.,2.] [3.,4.]] [[1.,2.] [3.,4.]]]
    belief_state = spzeros(10000, 1)
    belief_state[1,1] = 10
    belief_states = [belief_state belief_state]

    update!(e, uav_actions, utilities, belief_states)
    @test e.actions == reshape(uav_actions.actions, (2, 1))
    update!(e, uav_actions, utilities, belief_states)
    @test e.actions == [uav_actions.actions uav_actions.actions]
end

function test_update_state()
    e = Encounter()
    uav_state_own = UAVState(ones(Float64, 8)...)
    uav_state_int = UAVState(zeros(Float64, 8)...)
    s = State([uav_state_own; uav_state_int])
    update!(e, s)
    @test e.paths[1].x_coordinates == [1]
    @test e.paths[1].y_coordinates == [1]
    @test e.paths[2].x_coordinates == [0]
    @test e.paths[2].y_coordinates == [0]
    update!(e, s)
    @test e.paths[1].x_coordinates == [1, 1]
    @test e.paths[1].y_coordinates == [1, 1]
    @test e.paths[2].x_coordinates == [0, 0]
    @test e.paths[2].y_coordinates == [0, 0]
end

function main()
    # Environment tests
    test_constructor()

    # update! tests
    test_update_actions()
    test_update_state()
end

@time main()