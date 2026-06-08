# Benchmark energy-management MPC solvers against the stored high-accuracy baseline.
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using BenchmarkTools
using BlockArrays
using Distributions
using JSON3
using LinearAlgebra
using NNlib
using NPZ
using Plots
using Printf
using SparseArrays
using StaticArrays
using Statistics
using StatsPlots
using Base.Threads
using JuMP

import MathOptInterface as MOI
import ParametricOptInterface as POI


const tol::Float64 = 1e-2
const s_mb::Int = 24
const max_opt_gap::Float64 = 0.01

const ADMM_SAMPLES::Int = 300
const LME_ADMM_SAMPLES::Int = 300
const SPLIT_LME_ADMM_SAMPLES::Int = 300
const GC_FREQUENCY::Int = 10

const DATASET_DIR = joinpath(@__DIR__, "datasets")
const FIGURE_DIR = joinpath(@__DIR__, "media", "energy_management")

include("setup_em.jl")
include("preprocess.jl")
include("solver_em.jl")
include("admm_em.jl")
include("mel_admm_em.jl")


function benchmark_horizon(data::MPCData_eco)
    """Return the default initial state and forecast vectors used in benchmark runs.
    Args:
        data: Energy-management MPC data.

    Returns:
        Initial state, load forecast, and generation forecast.
    """
    load = Vector{Float64}(data.load_forecast[1:data.N])
    gen  = Vector{Float64}(data.gen_forecast[1:data.N])
    return data.x0, load, gen
end


function print_benchmark_header(label::String, samples::Int)
    """Print a consistent benchmark section header.
    Args:
        label: Solver or method name.
        samples: Number of timing samples.

    Returns:
        Nothing.
    """
    println("=============== Benchmarking $label: \033[1m #samples = $samples, N = $N, relative optimality gap = $max_opt_gap% \033[0m ===============")
    return nothing
end


function collect_solver_times(func_return_time, args::Tuple, n_sample::Int)
    """Collect solver-reported timing samples while preserving the last objective value.
    Args:
        func_return_time: Function returning `(solution, solve_time, objective_value)`.
        args: Arguments passed to `func_return_time`.
        n_sample: Number of benchmark samples.

    Returns:
        Timing samples after dropping the first warm-up run, and the last objective value.
    """
    time_arr = zeros(Float64, n_sample)
    objective_value = zero(Float64)

    for sample in 1:n_sample
        _, time_arr[sample], objective_value = func_return_time(args...)
        if sample % GC_FREQUENCY == 0
            GC.gc()
        end
    end

    return time_arr[2:end], objective_value
end


function report_timing(label::String, times::Vector{Float64})
    """Print mean and standard-deviation timing statistics in milliseconds.
    Args:
        label: Solver or method name.
        times: Solver-reported timing samples in seconds.

    Returns:
        Fitted Normal distribution of solver times in seconds.
    """
    stats = fit(Normal, times)
    @printf("%s time (mean ± σ): %5.3f ms ± %5.3f ms\n", label, stats.μ * 1000, stats.σ * 1000)
    return stats
end


function report_opt_gap(label::String, objective_value::Float64)
    """Print relative objective gap against the benchmark baseline.
    Args:
        label: Solver or method name.
        objective_value: Objective value returned by the solver.

    Returns:
        Nothing.
    """
    opt_gap = 100 * abs((Jopt - objective_value) / Jopt)
    @printf("opt_gap between %s objective and baseline: %5.3f%%\n", label, opt_gap)
    return nothing
end


function benchmark_solver(label::String, solver::Function, args::Tuple, samples::Int)
    """Run one solver benchmark section and return raw timing samples.
    Args:
        label: Solver or method name.
        solver: Callable benchmark target.
        args: Arguments passed to `solver`.
        samples: Number of timing samples.

    Returns:
        Timing samples in seconds, fitted timing distribution, and last objective value.
    """
    print_benchmark_header(label, samples)
    solver(args...; verbose = true)

    times, objective_value = collect_solver_times(solver, args, samples)
    stats = report_timing(label, times)
    report_opt_gap(label, objective_value)
    println()

    return times, stats, objective_value
end


function save_benchmark_plot(timing_ms::Dict{String, Vector{Float64}}, labels::Vector{String})
    """Save the timing box plot and raw benchmark timing arrays.
    Args:
        timing_ms: Mapping from solver labels to timing samples in milliseconds.
        labels: Plot and print order for solver labels.

    Returns:
        Nothing.
    """
    mkpath(DATASET_DIR)
    mkpath(FIGURE_DIR)

    bp = boxplot(timing_ms[labels[1]],
                 size = (450, 600),
                 label = labels[1],
                 framestyle = :box,
                 outliers = false,
                 xticks = (1:length(labels), fill("", length(labels))),
                 tickfont = 16,
                 guidefont = 16,
                 legendfont = 16)

    for label in labels[2:end]
        boxplot!(timing_ms[label], label = label, outliers = false)
    end

    savefig(bp, joinpath(FIGURE_DIR, string("eco_mpc-opt_gap=", max_opt_gap, ".pdf")))
    npzwrite(joinpath(DATASET_DIR, string("all_eco_mpc-opt_gap=", max_opt_gap, ".npz")), timing_ms)

    return nothing
end


function save_timeseries_plot(solution::Matrix{Float64}, data::MPCData_eco, path::String)
    """Save a four-panel time-series plot for the split LME-ADMM solution.
    Args:
        solution: Matrix whose first four rows contain m, u, p, and state trajectories.
        data: Energy-management MPC data.
        path: Output PDF path.

    Returns:
        Nothing.
    """
    time_grid = collect(0:data.dT:data.dT * (data.N - 1))

    plt = plot(layout = (2, 2), size = (900, 600))
    labels = ["(a)", "(b)", "(c)", "(d)"]
    units = ["kW", "kW", "kW", "%"]

    for i in 1:4
        series = i == 4 ? solution[i, :] .* 100 : solution[i, :]
        ylims_ = i == 3 ? (4, 6) : nothing

        plot!(plt[i],
              time_grid,
              series,
              xlabel = "Time (h)",
              ylabel = "",
              xlims = (0, data.dT * data.N),
              ylims = ylims_,
              xticks = 0:2:24,
              lw = 2.0,
              legend = false,
              grid = true,
              tickfont = font(12),
              guidefont = font(14))

        xmin, xmax = extrema(time_grid)
        ymin, ymax = ylims_ === nothing ? extrema(series) : ylims_
        xpos = xmin + 0.02 * (xmax - xmin)
        ypos = (ymin + ymax) / 2
        annotate!(plt[i], (xpos, ypos, text(labels[i], 14, :bold, :left)))

        ypos_unit = ymax + 0.05 * (ymax - ymin)
        annotate!(plt[i], (xpos, ypos_unit, text(units[i], 12, :left)))
    end

    mkpath(dirname(path))
    savefig(plt, path)
    println("Figure saved as '$path'")

    return nothing
end


function split_solver_with_objective(solver::Function, data::MPCData_eco)
    """Wrap split LME-ADMM so benchmark helpers receive an objective value.
    Args:
        solver: Split LME-ADMM solver returning `(solution, solve_time)`.
        data: Energy-management MPC data used to evaluate the objective.

    Returns:
        A solver returning `(solution, solve_time, objective_value)`.
    """
    return function wrapped_solver(init::Float64, load::Vector{Float64}, gen::Vector{Float64}, callback::Function; verbose::Bool = false)
        solution, solve_time = solver(init, load, gen, callback; verbose = verbose)
        return solution, solve_time, get_objective(data, solution)
    end
end


initial_state, load, gen = benchmark_horizon(mpc_data)

timing_labels = String[]
timing_data = Dict{String, Vector{Float64}}()

for (solver_name, samples, solver_tol) in [("Ipopt", 300, tol), ("MadNLP", 100, tol)]
        solver = mpc_eco_solver(solver_name, mpc_data, solver_tol)
        times, _, objective_value = benchmark_solver(solver_name, solver, (initial_state, load, gen), samples)
        timing_data[solver_name] = times .* 1000
        push!(timing_labels, solver_name)
        report_opt_gap(solver_name, objective_value)
end


prime_sol = prime_sol_struct("Ipopt", mpc_data)
aux_sol = aux_solver_eco("Clarabel", mpc_data)
admm_sol = ADMM_eco_iter(mpc_data, prime_sol, aux_sol)

admm_times, _, _ = benchmark_solver(
    "ADMM with Ipopt primal and Clarabel auxiliary",
    admm_sol,
    (initial_state, load, gen, ADMM_callback),
    ADMM_SAMPLES,
)
timing_data["ADMM"] = admm_times .* 1000
push!(timing_labels, "ADMM")

mgrad = gradient_struct(model, s_mb, dim)
aux_sol = aux_solver_eco("Clarabel", mpc_data)
ladmm_sol = LME_ADMM(mpc_data, mgrad, aux_sol)

ladmm_times, _, _ = benchmark_solver(
    "LME-ADMM",
    ladmm_sol,
    (initial_state, load, gen, ADMM_callback),
    LME_ADMM_SAMPLES,
)
timing_data["LME-ADMM"] = ladmm_times .* 1000
push!(timing_labels, "LME-ADMM")

mgrad = gradient_struct(model, s_mb, dim)
split_aux_sol = dynamics_projection(mpc_data)
split_sol = LME_ADMM_split(mpc_data, mgrad, split_aux_sol)
split_sol_with_objective = split_solver_with_objective(split_sol, mpc_data)

split_times, _, _ = benchmark_solver(
    "LME-ADMM with splitting",
    split_sol_with_objective,
    (initial_state, load, gen, sLME_ADMM_callback),
    SPLIT_LME_ADMM_SAMPLES,
)
timing_data["sLME-ADMM"] = split_times .* 1000
push!(timing_labels, "sLME-ADMM")

save_benchmark_plot(timing_data, timing_labels)

for label in timing_labels
    println("$label: ", median(timing_data[label]))
end

split_solution, _ = split_sol(initial_state, load, gen, sLME_ADMM_callback; verbose = true)
split_objective = get_objective(mpc_data, split_solution)
println("objective_value = $split_objective\n")
save_timeseries_plot(split_solution, mpc_data, joinpath(FIGURE_DIR, "bess_timeseries.pdf"))
