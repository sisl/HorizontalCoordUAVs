#=
Functions for building the environment from options and default values.
=#

using GridInterpolations
using JLD

### Simulation settings and constants ###

# These constants do not go in the options class because they 
# should essentially never change
const RANGES = [0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 6000, 8000, 10000, 12800]
const ACTIONS = deg2rad([-12, -6, 0, 6, 12, rad2deg(-1)])
const THETA_DIM = 37
const THETAS = collect(linspace(-pi, pi, THETA_DIM))
const BEARING_DIM = 37
const BEARINGS = collect(linspace(0, 2 * pi, BEARING_DIM))
const SPEED_DIM = 3
const SPEEDS = collect(linspace(35, 45, SPEED_DIM))
const RESPONDINGS = [0, 1]

### Helper functions for building the environment ###

# type for default simulation settings
type SimulationOptions
    # coordination table policy
    optimal_policy_filepath::ASCIIString
    
    # ownship options
    ownship_policy_filepath::ASCIIString
    policy_name::ASCIIString
    ownship_strategy::Function # {greedy_strategy, coordinated_strategy}
    ownship_action_selection::ASCIIString # {best*, worst*, average*}

    # intruder options
    intruder_policy_filepath::ASCIIString
    intruder_strategy::Function # {greedy_strategy, coordinated_strategy}
    intruder_action_selection::ASCIIString # {best*, worst*, average*}

    # environment options
    start_state_generator::Function
    use_noise::Bool
    random_seed::Int64
    gauss_sigma::Float64
    gauss_mu::Float64
    max_steps::Int64
    min_start_range_ratio::Float64
    max_start_range_ratio::Float64
    range_std::Float64
    initial_heading_offset::Float64
    initial_min_separation::Float64
    delayed_pilot_response::Bool

    # dynamics options 
    omega::Float64
    dt::Int64
    dt_step::Float64
    pid_b_std::Float64
    pid_b_std_coc::Float64

    function SimulationOptions()
        # optimal policy used for coordination table
        optimal_policy_filepath = "../../data/qvalue_tables/3.0_0.005_10000_10000.jld"

        # ownship options
        ownship_policy_filepath = "../../data/qvalue_tables/3.0_0.005_10000_10000.jld"

        policy_name = "solQ"
        ownship_strategy = coordinated_strategy # {greedy_strategy, coordinated_strategy}
        ownship_action_selection = "average" # {best, worst, average}
 
        # intruder options
        intruder_policy_filepath = "../../data/qvalue_tables/3.0_0.005_10000_10000.jld"
        intruder_strategy = coordinated_strategy
        intruder_action_selection = "average" # {best, worst, average}
        if ownship_strategy == greedy_strategy
            intruder_strategy = greedy_strategy
        end

        # environment options
        # generators: {start_towards_origin_encounter, start_head_on_encounter}
        start_state_generator = start_towards_origin_encounter 
        use_noise = true

        # for running on sherlock with multiple subtasks
        # get the id of this subtask to use as the seed
        local_id = 1
        try
            local_id = parse(Int64, ENV["SLURM_LOCALID"])
            # if you want to run on multiple nodes as well,
            # then use the environment variable 
            # SLURM_NODEID also, which gives the node
        catch e
            println("SLURM_LOCALID not set, are you running this using sbatch?")
        end
        random_seed = local_id

        gauss_sigma = deg2rad(2)
        gauss_mu = 0
        max_steps = 25
        min_start_range_ratio = 5 / 64 # gives min range of 2000
        max_start_range_ratio = 5 / 32 # gives max range of 2000
        range_std = 100.
        initial_heading_offset = pi / 32
        initial_min_separation = 153.
        delayed_pilot_response = true

        # dynamics options
        omega = .2
        dt = 5
        dt_step = 1.
        pid_b_std = deg2rad(3)
        pid_b_std_coc = deg2rad(5)

        # build the default options
        return new(optimal_policy_filepath, ownship_policy_filepath,
            policy_name, ownship_strategy, ownship_action_selection, 
            intruder_policy_filepath, intruder_strategy, intruder_action_selection, 
            start_state_generator, use_noise, random_seed, gauss_sigma, 
            gauss_mu, max_steps, min_start_range_ratio,
            max_start_range_ratio, range_std, initial_heading_offset, 
            initial_min_separation, delayed_pilot_response, 
            omega, dt, dt_step, pid_b_std, pid_b_std_coc)
    end
end

#=
Description:
Returns an enviornment object

Return Value:
- environment: an environment object
=#
function build_environment(; accept_command_line_options = true)
    # should instead gather options from cmd line
    options = SimulationOptions()
    if accept_command_line_options
        parse_command_line_options!(options)
    end
    
    # build uavs and dynamics
    uavs = build_uavs(options)
    dynamics = build_dynamics(options)

    # build actual environment
    max_steps = options.max_steps
    use_noise = options.use_noise
    gauss_sigma = options.gauss_sigma
    gauss_mu = options.gauss_mu
    start_state_generator = options.start_state_generator
    min_start_range_ratio = options.min_start_range_ratio
    max_start_range_ratio = options.max_start_range_ratio
    range_std = options.range_std
    initial_heading_offset = options.initial_heading_offset
    initial_min_separation = options.initial_min_separation
    delayed_pilot_response = options.delayed_pilot_response

    rng = MersenneTwister(options.random_seed)
    state_generator_rng = MersenneTwister(options.random_seed)
    e = Environment(max_steps, use_noise, uavs, gauss_sigma, 
                    gauss_mu, dynamics, start_state_generator, 
                    min_start_range_ratio, max_start_range_ratio, range_std,
                    initial_heading_offset, initial_min_separation, 
                    delayed_pilot_response, rng, state_generator_rng)
    return e
end

#=
Description:
Make joint actions given list of actions

Parameters:
- actions: list of float valued actions

Return Value:
- array of actions such that each row corresponds to a uav 
    and col to a joint action index, with the cell value being the action
=#
function get_joint_actions(actions)
    num_actions = length(actions)
    joint_actions = zeros(2, num_actions^2)
    i_actions = 1
    for ia2 = 1:num_actions
        for ia1 = 1:num_actions
            joint_actions[1, i_actions] = actions[ia1]
            joint_actions[2, i_actions] = actions[ia2]
            i_actions += 1
        end
    end
    return joint_actions
end

#=
Description:
Load in a policy from a filepath

Parameters:
- filepath: filepath to policy
- options: options type with info to build policy

Return Value:
- policy_table type object
=#
function load_policy(filepath::ASCIIString, options::SimulationOptions)
    q_table = load(filepath, options.policy_name)
    q_grid = RectangleGrid(RANGES, THETAS, BEARINGS, SPEEDS, SPEEDS, RESPONDINGS, RESPONDINGS)
    num_q_table_states = reduce(*, q_grid.cut_counts) + 1
    msg = "the q_table (rows = $(size(q_table, 1))) does not match the 
        q_grid (length = $num_q_table_states), which was created using constant values"
    @assert num_q_table_states == size(q_table, 1) msg
    policy = PolicyTable(q_table, q_grid, num_q_table_states)
    return policy
end

#=
Description:
Returns a list of uav objects

Return Value:
- uavs: list of uav objects
=#
function build_uavs(options::SimulationOptions)

    # alias this constant value 
    actions = ACTIONS

    # load ownship policy
    policy = load_policy(options.ownship_policy_filepath, options)

    # determine the joint actions
    joint_actions = get_joint_actions(actions)

    # build the coordination table
    optimal_policy = load(options.optimal_policy_filepath, options.policy_name) 
    c_table = make_coordiantion_table(optimal_policy, joint_actions)
    c_grid = RectangleGrid(RANGES, THETAS, BEARINGS, SPEEDS, SPEEDS, RESPONDINGS, RESPONDINGS)
    num_coord_states = reduce(*, c_grid.cut_counts)
    coord = CoordinationTable(c_table, c_grid, num_coord_states)

    # create the ownship
    ownship_id = 1
    ownship_uav = UAV(ownship_id, actions, joint_actions, policy, 
                    coord, options.ownship_strategy, options.ownship_action_selection)

    # create the intruder
    if options.intruder_policy_filepath == options.ownship_policy_filepath
        intruder_policy = policy
    else
        intruder_policy = load_policy(options.intruder_policy_filepath, options)
    end

    intruder_id = 2
    strategy = options.intruder_strategy
    intruder_uav = UAV(intruder_id, actions, joint_actions, intruder_policy, 
                    coord, strategy, options.intruder_action_selection)

    return [ownship_uav, intruder_uav]
end

#=
Description:
Returns a dynamics object

Return Value:
- dynamics: a dynamics object
=#
function build_dynamics(options::SimulationOptions)
    m = MersenneTwister(options.random_seed)
    omega = options.omega
    use_noise = options.use_noise
    dt = options.dt
    dt_step = options.dt_step
    pid_b_std = options.pid_b_std
    pid_b_std_coc = options.pid_b_std_coc
    d = Dynamics(dt, dt_step, omega, pid_b_std, pid_b_std_coc, use_noise, m)
    return d
end

#=
Description:
Parses command line options and replaces default values
in the options type with their values.

Side Effects:
- replaces default options with command line options if they exist
=#
function parse_command_line_options!(options::SimulationOptions)
    parsed_options = Dict()
    parsed_options["ownship_policy_filepath"] = nothing
    parsed_options["ownship_action_selection"] = nothing
    parsed_options["intruder_policy_filepath"] = nothing
    parsed_options["intruder_action_selection"] = nothing

    if length(ARGS) >= 1
        parsed_options["ownship_policy_filepath"] = ARGS[1]
    end

    if length(ARGS) >= 2
        parsed_options["ownship_action_selection"] = ARGS[2]
    end 

    if length(ARGS) >= 3
        parsed_options["intruder_policy_filepath"] = ARGS[3]
    end 

    if length(ARGS) >= 4
        parsed_options["intruder_action_selection"] = ARGS[4]
    end 

    # set the ownship_policy_filepath
    filepath = parsed_options["ownship_policy_filepath"]
    if filepath != nothing
        options.ownship_policy_filepath = filepath
    end

    # set the ownship_action_selection
    method_prefix = parsed_options["ownship_action_selection"]
    if method_prefix != nothing
        if method_prefix == "best" || method_prefix == "worst" || method_prefix == "average"
            method = method_prefix
            options.ownship_strategy = coordinated_strategy
        elseif method_prefix == "base"
            # method does not matter in this case
            method = "base"
            # set intruder to be greedy is ownship not coordinating
            parsed_options["intruder_action_selection"] = "base"
            options.ownship_strategy = greedy_strategy
        else
            throw(ArgumentError("invalid action selection method $(method_prefix)"))
        end
        options.ownship_action_selection = method
    end

    # set intruder filepath
    filepath = parsed_options["intruder_policy_filepath"]
    if filepath != nothing
        options.intruder_policy_filepath = filepath
    end

    # set intruder action selection
    method_prefix = parsed_options["intruder_action_selection"]
    if method_prefix != nothing
        if method_prefix == "best" || method_prefix == "worst" || method_prefix == "average"
            method = method_prefix
            options.intruder_strategy = coordinated_strategy
        elseif method_prefix == "base"
            # method does not matter in this case
            method = "base"
            options.intruder_strategy = greedy_strategy
        else
            throw(ArgumentError("invalid action selection method $(method_prefix)"))
        end
        options.intruder_action_selection = method
    end
end

