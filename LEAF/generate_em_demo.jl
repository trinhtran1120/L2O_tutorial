module EnergyManagementDemo

using Base.Threads
using JSON3
using LinearAlgebra
using NPZ
using Printf
using SparseArrays
using StaticArrays

import MathOptInterface as MOI
import ParametricOptInterface as POI

include("setup_em.jl")
include("preprocess.jl")
include("solver_em.jl")
include("admm_em.jl")

const DATASET_DIR = joinpath(@__DIR__, "datasets")

function generate_demo_training_data(; m::Int = 1, n::Int = 2, tol::Float64 = 1e-2)
    mkpath(DATASET_DIR)

    solver_name = "Ipopt"
    aux_solver_name = "Clarabel"

    mpc_data = energy_mag()
    data_train = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[])

    # Build the ADMM subproblem solvers once.
    prime_sol = prime_solver_eco_data(solver_name, mpc_data)
    aux_sol = aux_solver_eco_data(aux_solver_name, mpc_data)
    admm_sol = ADMM_eco_iter_data(mpc_data, prime_sol, aux_sol; max_iter = n, tol = tol)

    # Collect a tiny training set from a few initial battery states.
    train_pool = [1/2, 2/3, 3/4]

    for x0 in train_pool
        println("Collecting training data with initial SOC = $(round(100x0, digits=1))%")
        _, J_ADMM = admm_sol(data_train, x0; verbose = true)
        @printf("J_ADMM = %5.3f\n", J_ADMM)
    end

    # Save in the same input/enve/grad format used by the full generator.
    data = Dict(
        "input" => reduce(hcat, data_train["input"]),
        "grad" => reduce(hcat, data_train["grad"]),
        "rho" => mpc_data.rho,
        "enve" => data_train["env"],
    )

    output_path = joinpath(DATASET_DIR, "eco_mpc-demo-train.npz")
    npzwrite(output_path, data)
    @printf("Collected %d training data points\n", length(data_train["input"]))
    println("Saved demo dataset to $(output_path)")

    return data
end

end


# Generate the demo training datay
using .EnergyManagementDemo: generate_demo_training_data
