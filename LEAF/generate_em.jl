# Generate training and testing datasets for the energy-management MPC example.
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JSON3
using LinearAlgebra
using NPZ
using Printf
using SparseArrays
using StaticArrays
using Base.Threads

import MathOptInterface as MOI
import ParametricOptInterface as POI

const max_iter::Int = 1000
const DATASET_DIR = joinpath(@__DIR__, "datasets")

mkpath(DATASET_DIR)

include("setup_em.jl")
include("preprocess.jl")
include("solver_em.jl")
include("admm_em.jl")

solver_name = "Ipopt"
aux_solver_name = "Clarabel"

mpc_data = energy_mag()
cost_func = mpc_data.cost_func

# Dataset dictionaries store ADMM/proximal inputs, envelope values, and gradient targets.
data_train = Dict("input" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[])
data_test  = Dict("input" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[])

# Build the solvers once; each sampled initial state reuses the same model structure.
mpc_eco_sol = mpc_eco_solver(solver_name, mpc_data, 1e-6)
prime_sol   = prime_solver_eco_data(solver_name, mpc_data)
aux_sol     = aux_solver_eco_data(aux_solver_name, mpc_data)
admm_sol    = ADMM_eco_iter_data(mpc_data, prime_sol, aux_sol; tol=1e-3)

train_pool = [1/2, 2/3, 3/4]
test_pool  = [3/5]

opt_sol  = nothing
opt_ADMM = nothing

load = mpc_data.load_forecast[1:mpc_data.N]
gen  = mpc_data.gen_forecast[1:mpc_data.N]

# Collect training data from multiple initial states of charge.
for x0 in train_pool
    global opt_sol, opt_ADMM

    println("================ Collecting training data with initial SOC of BESS = $(round(100*x0, digits=1))% ================")
    println("---------------- $solver_name ----------------")

    opt_sol, solve_time, J_opt = mpc_eco_sol(x0, load, gen)
    @printf("J_opt = %5.3f, solving time = %5.2f ms\n", J_opt, solve_time*1000) 
    
    println("---------------- ADMM ----------------")
    opt_ADMM, J_ADMM = admm_sol(data_train, x0; verbose = true)
    @printf("J_ADMM = %5.3f\n", J_ADMM) 
    @printf("ΔJ/J = %5.3f%%\n", abs(J_opt - J_ADMM) / abs(J_opt) * 100)
    @printf("max|opt_sol  - opt_ADMM| = %5.3f\n\n", maximum(abs.(opt_ADMM - opt_sol)))
end

@printf("Collected %4d training data points \n\n", length(data_train["input"]))
data = Dict("input" => reduce(hcat, data_train["input"]), "grad" => reduce(hcat, data_train["grad"]), "rho" => mpc_data.rho, "enve" => data_train["env"])
npzwrite(joinpath(DATASET_DIR, string(typeof(cost_func), "-rho=", mpc_data.rho, "-train", ".npz")), data)

# Collect testing data from held-out initial states of charge.
for x0 in test_pool
    println("================ Collecting testing data with initial SOC of BESS = $(round(100*x0, digits=1))% ================")
    opt_ADMM, J_ADMM = admm_sol(data_test, x0; verbose = true)
    @printf("J_ADMM = %5.3f\n", J_ADMM) 
end

@printf("Collected %4d testing data points \n\n", length(data_test["input"]))
data = Dict("input" => reduce(hcat, data_test["input"]), "grad" => reduce(hcat, data_test["grad"]), "rho" => mpc_data.rho, "enve" => data_test["env"])
npzwrite(joinpath(DATASET_DIR, string(typeof(cost_func), "-rho=", mpc_data.rho, "-test", ".npz")), data)
