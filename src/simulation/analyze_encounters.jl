#=
Functions for analyzing encounters that result from running simulations.
=#

using PGFPlots
using PyPlot: show, close, scatter
using TikzPictures

# where to output 
const OUTPUT_DIRECTORY = "../../data/plots/"
const CONFLICT_DISTANCE = 153

# these map actions to values used for plotting
const ACTION_NAMES = Dict(-1.0 => "coc", 
                deg2rad(12) => "left5",
                deg2rad(6) => "left2.5",
                deg2rad(0) => "center",
                deg2rad(-12) => "right5",
                deg2rad(-6) => "right2.5",)

const ACTION_COLORS = Dict(-1.0 => .5,
                deg2rad(12) => .1,
                deg2rad(6) => .1,
                deg2rad(0) => .5,
                deg2rad(-12) => .9,
                deg2rad(-6) => .9)

#=
Description:
Plot the x and y coordinates of both planes in an encounter.

Parameters:
- encounter: the encounter to plot
- idx: the number of this encounter in the set
- nmac_only: only plot if this enounter was an nmac if true
- as_tex: if true, output the plots as tex files for including in papers
=#
function plot_encounter_paths(encounter::Encounter, idx::Int64; 
            nmac_only::Bool = false, as_tex::Bool = false)
    if nmac_only && encounter.nmac == false
        return
    end

    a = PGFPlots.Axis(
            [
                Plots.Linear(encounter.paths[1].x_coordinates, 
                            encounter.paths[1].y_coordinates, 
                            mark="none", 
                            style="blue,->,smooth,thick"), 

                Plots.Linear(encounter.paths[2].x_coordinates, 
                            encounter.paths[2].y_coordinates, 
                            mark="none", 
                            style="red,->,smooth,thick"),

                Plots.Node(L"\bullet", encounter.paths[1].x_coordinates[1],
                         encounter.paths[1].y_coordinates[1], style="blue"),
                Plots.Node(L"\bullet", encounter.paths[2].x_coordinates[1],
                         encounter.paths[2].y_coordinates[1], style="red"),
            ],
        xlabel="x (m)", ylabel="y (m)", width="12cm", height="12cm", 
        style="axis equal,xtick={-4000,-2000,...,4000}, ytick={-4000,-2000,...,4000}",
        xmin=-4000, xmax=4000, ymin=-4000, ymax=4000,
        axisEqualImage=true
    )
    ext = as_tex ? "tex" : "pdf"
    TikzPictures.save(string(OUTPUT_DIRECTORY, "paths_plot$(idx).$(ext)"), a)
end

#=
Description:
Plot the x and y coordinates of both planes in an encounter plus color code with actions

Parameters:
- encounter: the encounter to plot
=#
function plot_encounter_actions(encounter::Encounter, idx::Int64; 
            nmac_only::Bool = false, as_tex::Bool = false)
    if nmac_only && encounter.nmac == false
        return
    end

    colors = Array(Any, size(encounter.actions))
    for i in 1:size(colors, 1)
        for j in 1:size(colors, 2)
            colors[i, j] = ACTION_NAMES[encounter.actions[i, j]]
        end
    end
    colors = hcat(colors, [-1; -1])

    sc1 = "{left5={blue},left2.5={cyan},center={draw=black},right5={red},right2.5={orange},coc={green}}"
    sc2 = "{left5={mark=triangle*,blue},left2.5={mark=triangle*,cyan},center={mark=triangle*,draw=black},right5={mark=triangle*,red},right2.5={mark=triangle*,orange},coc={mark=triangle*,green}}"

    p1 = Plots.Scatter(encounter.paths[1].x_coordinates, 
                       encounter.paths[1].y_coordinates,
                       vec(colors[1, :]),
                       scatterClasses=sc1,
                       markSize=1,
                       )
    p2 = Plots.Scatter(encounter.paths[2].x_coordinates, 
                       encounter.paths[2].y_coordinates,
                       vec(colors[2, :]),
                       scatterClasses=sc2,
                       markSize=1)
    a = PGFPlots.Axis([p1, p2], 
            xlabel="x (m)", 
            ylabel="y (m)", 
            width="24cm", 
            height="24cm", 
            style="axis equal,xtick={-4000,-2000,...,4000}, ytick={-4000,-2000,...,4000}",
            xmin=-4000, xmax=4000, ymin=-4000, ymax=4000,
            axisEqualImage=true
        )
    ext = as_tex ? "tex" : "pdf"
    TikzPictures.save(string(OUTPUT_DIRECTORY, "action_plot$(idx).$(ext)"), a)
end

#=
Description:
This function steps through each timestep of the encounter. 
It displays the current state of the encounter and also prints 
out information to the terminal.

Parameters:
- encounter: the encounter to step through
=#
function step_through(encounter::Encounter)
    colors = Array(Any, size(encounter.actions))
    for i in 1:size(colors, 1)
        for j in 1:size(colors, 2)
            colors[i, j] = ACTION_COLORS[encounter.actions[i, j]]
        end
    end
    colors = hcat(colors, [ACTION_COLORS[-1]; ACTION_COLORS[-1]])

    for idx in 1:size(encounter.actions, 2)
        scatter(encounter.paths[1].x_coordinates[1:idx], 
                encounter.paths[1].y_coordinates[1:idx],
                s=400,
                marker="*")
        scatter(encounter.paths[2].x_coordinates[1:idx], 
                encounter.paths[2].y_coordinates[1:idx],
                s=400,
                marker=".")

        println("########## $idx")
        println("########## responding")
        println(encounter.states[idx].responding_states)
        println("########## state")
        println(encounter.states[idx])
        println(encounter.states[idx].uav_states[1])
        println(encounter.states[idx].uav_states[2])
        println("########## polar state")
        println(to_polar(encounter.states[idx]))
        println("########## min horizontal distance")
        println(encounter.min_horizontal_dists[idx])
        println("########## belief state ownship")
        println(size(encounter.belief_states))
        println(encounter.belief_states[idx])
        println("########## utilities")
        println(encounter.utilities[idx])
        println("########## indmax")
        println(indmax(encounter.utilities[idx][1, :]))
        println("########## actions")
        println(encounter.actions[:, idx])
        println("########## signals")
        println(encounter.signals[:, idx])
        println("\n")

        show()
        resp = lowercase(readline(STDIN))
        if resp == "q" || resp == "quit"
            break
        end
        close()
    end

end

#=
Description:
Computes the alert and conflict metrics for an encounter

Parameters:
- encounter: the encounter to compute metrics for

Return Values:
- num_alerts: number of non COC actions
- in_conflicts: {0, 1} whether ther was ever a conflict
- conflict_ratio: ratio of states in conflict
=#
function compute_metrics(encounter::Encounter)

    # the values to compute
    num_alerts = 0
    num_conflicts = 0

    for action in encounter.actions
        if action != COC_ACTION
            num_alerts += 1
        end
    end

    for horizontal_distance in encounter.min_horizontal_dists
        if horizontal_distance < CONFLICT_DISTANCE
            num_conflicts += 1
        end
    end

    # in_conflict is whether there was ever a conflict
    in_conflict = 0
    if num_conflicts > 0
        in_conflict = 1
    end

    conflict_ratio = num_conflicts / length(encounter.states)

    return num_alerts, in_conflict, conflict_ratio
end

#=
Description:
Summarizes the statistics of a list of encounters.

Parameters:
- encounters: list of encounter objects
- verbose: how to print results, 0 for batch mode

Side Effects:
- flips nmac bit on each encounter if one occurred
=#
function summarize_encounters!(encounters::Vector{Encounter}; verbose::Int64 = 0)

    alerts = 0
    conflicts = 0
    conflict_ratio_sum = 0
    for e in encounters
        num_alerts, in_conflict, conflict_ratio = compute_metrics(e)

        if in_conflict == 1
            e.nmac = true
        end

        alerts += num_alerts
        conflicts += in_conflict
        conflict_ratio_sum += conflict_ratio
    end
    if verbose == 1
        println("Number of encounters simulated: $(length(encounters))")
        println("Total Alerts: $(alerts)")
        println("Average Alerts: $(alerts / length(encounters))")
        println("Total Conflicts: $(conflicts)")
        println("Average Conflicts: $(conflicts / length(encounters))")
        println("Fraction of States in Conflict: $(conflict_ratio_sum / length(encounters))")
    else
        # simple printed output used for automated stat collection by 
        # a separate script
        println("$(length(encounters)) $(conflicts) $(alerts)")
    end
end

#=
Description:
Main entry point for running analysis on the simulated encounters

Parameters:
- encounters: the encounters to analyze
=#
function analyze(encounters::Vector{Encounter})
    summarize_encounters!(encounters, verbose = 1)
    for (idx, encounter) in enumerate(encounters)
        plot_encounter_actions(encounter, idx, nmac_only = false, as_tex = false)
        plot_encounter_paths(encounter, idx, nmac_only = false, as_tex = false)
    end
    step_through(encounters[1])
end