# tests for the policy and coord tables types and related functions.

using Base.Test
using GridInterpolations

push!(LOAD_PATH, ".")
include("includes.jl")

function get_test_policy()
    q_table = zeros(4^7, 36)
    range = collect(1:4)
    q_grid = RectangleGrid(range, range, range, range, range, range, range)
    policy = PolicyTable(q_table, q_grid, 4^7)
end

function get_test_coordination_table()
    c_table = zeros(4^7)
    c_table[1] = 1
    c_table[2] = -1
    range = collect(1:4)
    c_grid = RectangleGrid(range, range, range, range, range, range, range)
    coordination_table = CoordinationTable(c_table, c_grid, 4^7)
end

# policy table tests
function test_policy_table_constructor()
    get_test_policy()
end

function test_get_utilities()
    policy = get_test_policy()
    policy.q_table[1, 1] = 10
    policy.q_table[2, 2] = 9
    belief = spzeros(policy.num_states, 1)
    belief[1] = .5
    belief[2] = .5

    utilities = get_utilities(policy, belief)
    expected_utilities = zeros(size(policy.q_table, 2))
    expected_utilities[1] = .5 * 10
    expected_utilities[2] = .5 * 9
    @test utilities == expected_utilities
end

function test_make_coordination_table()
    q_table = zeros(2, 36)
    q_table[1, 1] = 1
    q_table[2, 36] = 2
    q_table[2, 4] = 1
    joint_actions = get_joint_actions(ACTIONS)
    c_table = make_coordiantion_table(q_table, joint_actions)
    @test c_table[1] == 1
end

function test_get_sense_action_parameters()
    a1 = -10.
    a2 = +10.
    sense = get_action_sense(a1, a2)
    @test sense == :different_sense
    a1 = 0.
    sense = get_action_sense(a1, a2)
    @test sense == :neither_sense
    a1 = +10.
    sense = get_action_sense(a1, a2)
    @test sense == :same_sense
end

function test_get_sense_coord_table_parameters()
    c_table = get_test_coordination_table()
    belief = spzeros(c_table.num_states, 1)
    belief[1] = .6
    belief[2] = .4
    sense = get_coordination_sense(c_table, belief)
    @test sense == :same_sense

    belief[1] = .4
    belief[2] = .6
    sense = get_coordination_sense(c_table, belief)
    @test sense == :different_sense

    belief[1] = .3
    belief[2] = .3
    belief[3] = .4
    sense = get_coordination_sense(c_table, belief)
    @test sense == :neither_sense
end

function main()
    # policy tests
    test_policy_table_constructor()
    test_get_utilities()

    # coord tests
    test_make_coordination_table()
    test_get_sense_action_parameters()
    test_get_sense_coord_table_parameters()
end

@time main()