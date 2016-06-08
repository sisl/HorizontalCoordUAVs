# tests for the environment type and related functions.

using Base.Test
using PyPlot

push!(LOAD_PATH, ".")
include("includes.jl")

function get_test_uavs()
    actions = [-1., 1., 0.]
    joint_actions = ones(3, 3)
    joint_actions[:, 1] = -1

    r = collect(1:10)
    q_table = zeros(10^7, size(joint_actions, 2))
    q_grid = RectangleGrid(r, r, r, r, r, r, r)
    policy = PolicyTable(q_table, q_grid, 10^7)

    c_table = zeros(10, 10, 10, 10, 10)
    c_grid = RectangleGrid(r * 100, r, r, r, r)
    coord = CoordinationTable(c_table, c_grid, 10^5)

    strategy = coordinated_strategy
    act_select = "best"

    uav1 = UAV(1, actions, joint_actions, policy, coord, strategy, act_select)
    uav2 = UAV(2, actions, joint_actions, policy, coord, strategy, act_select)

    return [uav1, uav2]
end

function get_test_env()
    max_steps = 100
    use_noise = true
    uavs = get_test_uavs()
    gauss_sigma = 1.0
    gauss_mu = 0.0
    d = Dynamics(3, 1., 4., 5., 5., false, MersenneTwister(1))
    start_state_generator = start_head_on_encounter
    rng = MersenneTwister(1)
    e = Environment(100, use_noise, uavs, gauss_sigma, gauss_mu, d,
                    start_state_generator, 1/4, 1/3, 100., pi / 32, 153., true, rng, rng)
    return e
end

# Environment tests
function test_constructor()
    get_test_env()
end

function test_get_uav_actions()
    env = get_test_env()
    uav_state_own = UAVState(ones(Float64, 8)...)
    uav_state_int = UAVState(zeros(Float64, 8)...)
    s = State([uav_state_own; uav_state_int])
    enc = Encounter()
    actions = get_uav_actions!(env, s, enc)
    @test length(actions.actions) == s.num_uavs
end

function test_step()
    env = get_test_env()
    uav_state_own = UAVState(ones(Float64, 8)...)
    uav_state_int = UAVState(ones(Float64, 8)...)
    s = State([uav_state_own; uav_state_int])
    uav_actions = UAVActions(2)
    uav_actions.actions = [deg2rad(10); deg2rad(10)]
    enc = Encounter()
    next_state = step!(env, s, uav_actions, enc)
end

function test_add_noise()
    env = get_test_env()
    actions = [0., 0.]
    add_noise!(env, actions)
    @test actions != [0., 0.]
end

function test_start_encounter_head_on()
    env = get_test_env()
    env.start_state_generator = start_head_on_encounter
    num_runs = 500
    int_xs = zeros(num_runs)
    int_ys = zeros(num_runs)
    angles = zeros(num_runs)
    for idx in 1:num_runs
        state = start_encounter(env)
        int_xs[idx] = state.uav_states[2].x
        int_ys[idx] = state.uav_states[2].y
        angles[idx] = state.uav_states[2].psi
    end
    scatter(int_xs, int_ys, c=angles)
    title("position and heading of intruder uav at start of episode")
    if SHOW_PLOTS
        show()
    end
end

function test_start_encounter_is_deterministic()
    env_1 = get_test_env()
    state_1 = start_encounter(env_1)
    env_2 = get_test_env()
    state_2 = start_encounter(env_2)
    env_3 = get_test_env()
    state_3 = start_encounter(env_3)
    @test state_1 == state_2 == state_3
    state_12 = start_encounter(env_1)
    state_22 = start_encounter(env_2)
    @test state_12 == state_22
end

function test_start_encounter_towards_origin()
    env = build_environment()
    env.start_state_generator = start_towards_origin_encounter
    num_runs = 300
    own_xs = zeros(num_runs)
    own_ys = zeros(num_runs)
    int_xs = zeros(num_runs)
    int_ys = zeros(num_runs)
    for idx in 1:num_runs
        state = start_encounter(env)
        own_xs[idx] = state.uav_states[1].x
        own_ys[idx] = state.uav_states[1].y
        int_xs[idx] = state.uav_states[2].x
        int_ys[idx] = state.uav_states[2].y

        # xlim([-2000, 2000])
        # ylim([-2000, 2000])
        # scatter(own_xs[idx], own_ys[idx], c="blue")
        # scatter(int_xs[idx], int_ys[idx], c="red")
        # show()
        # readline()
        # close()
    end
    scatter(own_xs, own_ys, c="blue")
    scatter(int_xs, int_ys, c="red")
    title("position and heading of uavs at start of episode")
    if SHOW_PLOTS
        show()
    end
end

function plot_start_states(env, num_runs, c1, c2, extra_random)
    if extra_random
        rand(env.rng, 1:10)
    end

    own_xs = zeros(num_runs)
    own_ys = zeros(num_runs)
    int_xs = zeros(num_runs)
    int_ys = zeros(num_runs)
    for idx in 1:num_runs
        state = start_encounter(env)
        own_xs[idx] = state.uav_states[1].x
        own_ys[idx] = state.uav_states[1].y
        int_xs[idx] = state.uav_states[2].x
        int_ys[idx] = state.uav_states[2].y
    end
    scatter(own_xs, own_ys, c=c1)
    scatter(int_xs, int_ys, c=c2)
    title("position and heading of uavs at start of episode")

end

function test_start_encounter_towards_origin_coordination_vs_greedy()
    num_runs = 100

    env = build_environment()
    env.start_state_generator = start_towards_origin_encounter
    env.uavs[1].strategy = coordinated_strategy
    plot_start_states(env, num_runs, "blue", "red", false)

    env = build_environment()
    env.start_state_generator = start_towards_origin_encounter
    env.uavs[1].strategy = greedy_strategy
    plot_start_states(env, num_runs, "green", "purple", true)

    if SHOW_PLOTS
        show()
    end
    
end

function main()
    # Environment tests
    test_constructor()

    # uav and dynamics tests
    test_get_uav_actions()
    test_step()
    test_add_noise()
    test_start_encounter_head_on()
    test_start_encounter_is_deterministic()
    test_start_encounter_towards_origin()
    test_start_encounter_towards_origin_coordination_vs_greedy()

end

@time main()