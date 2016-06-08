# tests for strategy functions.

using Base.Test

push!(LOAD_PATH, ".")
include("includes.jl")

function get_test_state()
    uav_state_own = UAVState(100., -10., 0., 0., 1., 0., 0., 0.)
    uav_state_int = UAVState(0., 10., 0., 0., 1., 0., 0., 0.)
    s = State([uav_state_own; uav_state_int])
    s.responding_states = [1,1]
    return s
end

function test_modify_utilities()
    uav = get_real_uav()
    state = get_test_state()
    # second signal doesn't matter here
    state.signals = [:dont_turn_left, :dont_turn_right]
    utilities = zeros(size(uav.joint_actions, 2))
    # intruder signalling to take same sense
    mod_utilities = modify_utilities(uav, state, utilities)
    exp_utilities = zeros(length(uav.joint_actions[1, :]))
    for aidx in 1:size(uav.joint_actions, 2)
        if sign(uav.joint_actions[1, aidx]) == 1
            exp_utilities[aidx] = -Inf
        end
    end
    @test mod_utilities == exp_utilities
end

function test_select_action_and_signals_best()

    environment = build_environment()
    environment.uavs[1].action_selection_method = "best"
    environment.uavs[2].action_selection_method = "best"

    no_prev_signals = [:no_signal, :no_signal]

    uav = environment.uavs[1]
    utilities = zeros(size(uav.joint_actions, 2))
    utilities[1] = 1
    sense = :same_sense
    action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
    @test signals[1] == signals[2]
    @test action == -deg2rad(12)

    sense = :different_sense
    utilities = zeros(size(uav.joint_actions, 2))
    utilities[3] = 1
    action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
    @test signals[1] != signals[2]
    @test action == 0.0

    uav = environment.uavs[2]
    m = MersenneTwister(1)
    utilities = rand(m, size(uav.joint_actions, 2))
    sense = :neither_sense
    action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
    best_action_idx = indmax(utilities)
    best_action = uav.joint_actions[uav.uav_id, best_action_idx]
    @test action == best_action
    @test signals == [:no_signal, :no_signal]

    uav = environment.uavs[2]
    utilities = zeros(size(uav.joint_actions, 2))
    utilities[21] = 10
    sense = :same_sense
    action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
    @test action == deg2rad(6)
    @test signals[1] == signals[2]
    uav = environment.uavs[1]
    action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
    @test action == 0.0
    @test signals[1] == signals[2]

    utilities = collect(1.:36.)
    sense = :same_sense
    action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
    @test action == COC_ACTION
    @test signals[1] == signals[2]
    @test signals[1] == :no_signal

end

function test_select_action_and_signals_worst()
    environment = build_environment()
    environment.uavs[1].action_selection_method = "worst"
    environment.uavs[2].action_selection_method = "worst"
    uav = environment.uavs[1]
    utilities = ones(size(uav.joint_actions, 2)) * -1
    utilities[[6, 12, 18, 24, 30, 36]] = 1
    sense = :same_sense
    action, signals = select_action_and_signals(uav, utilities, sense, [:no_signal, :no_signal])
    @test action == COC_ACTION
    @test signals[1] == :no_signal
    @test signals[1] == signals[2]

    uav = environment.uavs[2]
    utilities[30:end] = 1
    action, signals = select_action_and_signals(uav, utilities, sense, [:no_signal, :no_signal])
    @test action == COC_ACTION
    @test signals[1] == :no_signal
    @test signals[1] == signals[2]

    uav = environment.uavs[1]
    utilities = ones(size(uav.joint_actions, 2)) * -1
    utilities[[1, 7, 13, 19, 25, 31]] = 1
    utilities[1] = -100
    action, signals = select_action_and_signals(uav, utilities, sense, [:no_signal, :no_signal])
    @test action != -deg2rad(12)

    utilities = ones(size(uav.joint_actions, 2))
    utilities[32:end] = -1
    action, signals = select_action_and_signals(uav, utilities, sense, [:no_signal, :no_signal])
    @test action == -deg2rad(12)
    @test signals[1] == :dont_turn_left
    @test signals[2] == :dont_turn_left

    uav = environment.uavs[1]
    utilities = ones(size(uav.joint_actions, 2))
    state = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
    state.signals = [:dont_turn_left, :dont_turn_left]
    utilities = modify_utilities(uav, state, utilities)
    action, signals = select_action_and_signals(uav, utilities, sense, state.signals)
    @test action == 0.0
    @test signals[1] == :dont_turn_left
    @test signals[2] == :dont_turn_left

    uav = environment.uavs[1]
    utilities = ones(size(uav.joint_actions, 2))
    state = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
    state.signals = [:dont_turn_right, :dont_turn_right]
    utilities = modify_utilities(uav, state, utilities)
    utilities[[3, 9, 15, 21, 27, 33]] = -1
    utilities[[4, 10, 16, 22, 28, 34]] = -1
    utilities[[6, 12, 18, 24, 30, 36]] = -1
    action, signals = select_action_and_signals(uav, utilities, sense, state.signals)
    @test action == deg2rad(12)
    @test signals[1] == :dont_turn_right
    @test signals[2] == :dont_turn_right

    uav = environment.uavs[1]
    utilities = ones(size(uav.joint_actions, 2))
    sense = :different_sense
    state = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
    state.signals = [:dont_turn_right, :dont_turn_right]
    utilities = modify_utilities(uav, state, utilities)
    utilities[[3, 9, 15, 21, 27, 33]] = -1
    utilities[[4, 10, 16, 22, 28, 34]] = -1
    utilities[[6, 12, 18, 24, 30, 36]] = -1
    action, signals = select_action_and_signals(uav, utilities, sense, state.signals)
    @test action == deg2rad(12)
    @test signals[1] == :dont_turn_right
    @test signals[2] == :dont_turn_left
end

function test_select_action_and_signals_average()
    environment = build_environment()
    environment.uavs[1].action_selection_method = "average"
    environment.uavs[2].action_selection_method = "average"
    sense = :same_sense
    uav = environment.uavs[1]
    utilities = ones(size(uav.joint_actions, 2))
    state = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
    state.signals = [:dont_turn_right, :dont_turn_right]
    utilities = modify_utilities(uav, state, utilities)
    utilities[[3, 9, 15, 21, 27, 33]] = -1
    utilities[[4, 10, 16, 22, 28, 34]] = -.5
    utilities[[6, 12, 18, 24, 30, 36]] = -1
    utilities[17] = -10000
    action, signals = select_action_and_signals(uav, utilities, sense, state.signals)
    @test action == deg2rad(6)
    @test signals[1] == :dont_turn_right
    @test signals[2] == :dont_turn_right
end

function main()
    test_modify_utilities()
    test_select_action_and_signals_best()
    test_select_action_and_signals_worst()
    test_select_action_and_signals_average()
end

@time main()