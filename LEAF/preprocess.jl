include("utils.jl")

rho, mp = load_model(joinpath(@__DIR__, "model", "neco_mpc-rho=1.json"))

model = ICNN(
    mp.U[1], 
    mp.b[1],
    [ICNN_Layer(mp.U[i], mp.W[i], mp.b[i]) for i in 2:length(mp.U)],
    mp.v, 
    mp.a,
    mp.c)

mpc_data  = energy_mag()
N   = mpc_data.N
dim = mpc_data.dim

Jopt::Float64 = 36479.1
if N == 192
    Jopt = 73117.5
end

function Ipopt_callback_BM(
   alg_mod::Cint,
   iter_count::Cint,
   obj_value::Float64,
   inf_pr::Float64,
   inf_du::Float64,
   mu::Float64,
   d_norm::Float64,
   regularization_size::Float64,
   alpha_du::Float64,
   alpha_pr::Float64,
   ls_trials::Cint,
)
    rel_opt_gap = 100abs(Jopt - obj_value)/Jopt
    stop = (rel_opt_gap < max_opt_gap)

    return !stop #False means running, True means stopping
end

@kwdef mutable struct callback_struct
    rel_opt_gap::Vector{Float64} = Float64[]
    n_iter::Vector{Int} = Int[]
end


function ADMM_callback(
    z::Matrix{Float64},
    w::Matrix{Float64},
    α::Matrix{Float64},
    iter::Int,
    J::Float64,
    sol_time_upto_iter::Float64
)
    opt_gap = 100abs(J - Jopt)/Jopt #Terminate by optimality gap

    return opt_gap < max_opt_gap
end

function sLME_ADMM_callback(
    z::Matrix{Float64},
    w::Matrix{Float64},
    α::Matrix{Float64},
    v::Matrix{Float64},
    β::Matrix{Float64},
    iter::Int,
    J::Float64
)
    opt_gap = 100abs(J - Jopt)/Jopt + 1e9norm(w .- v, Inf) #Terminate by optimality gap

    return opt_gap < max_opt_gap
end


function ADMM_callback_iter(
    z::Matrix{Float64},
    w::Matrix{Float64},
    α::Matrix{Float64},
    iter::Int,
    J::Float64,
    total_time::Float64,
    cbs::callback_struct
)
    push!(cbs.rel_opt_gap, 100abs(J - Jopt)/Jopt)
    push!(cbs.n_iter, iter)

    return iter >= LADMM_n_iter
end

function sADMM_callback_iter(
    z::Matrix{Float64},
    w::Matrix{Float64},
    α::Matrix{Float64},
    v::Matrix{Float64},
    β::Matrix{Float64},
    iter::Int,
    J::Float64,
    cbs::callback_struct
)
    push!(cbs.rel_opt_gap, 100abs(J - Jopt)/Jopt)
    push!(cbs.n_iter, iter)
    return iter >= sLADMM_n_iter
end


function Ipopt_callback_iter(
    alg_mod::Cint,
    iter_count::Cint,
    obj_value::Float64,
    inf_pr::Float64,
    inf_du::Float64,
    mu::Float64,
    d_norm::Float64,
    regularization_size::Float64,
    alpha_du::Float64,
    alpha_pr::Float64,
    ls_trials::Cint,
    cbs::callback_struct
    )
    push!(cbs.rel_opt_gap, 100abs(Jopt - obj_value)/Jopt)
    push!(cbs.n_iter, iter_count)

    return (iter_count < Ipopt_n_iter)
end

function pick_solver(name, tol::Float64 = 1e-6, cbs::Union{Nothing, callback_struct} = nothing)
    #All solvers should share the same tolerance for stopping criteria
    str = lowercase(name)

    if str == "ipopt"      #for LP, QP, NLP
        model = Model(Ipopt.Optimizer)
        set_optimizer_attribute(model, "sb", "yes")
        set_optimizer_attribute(model, "print_level", 0)
        if tol >= 1e-3 # high tol implies low accuracy,
            set_optimizer_attribute(model, "nlp_scaling_method", "none") #set none scaling method
            set_optimizer_attribute(model, "tol",  1e-4) 
            MOI.set(model, Ipopt.CallbackFunction(), Ipopt_callback_BM) #set termination depends only on Ipopt_callback
        else
            set_optimizer_attribute(model, "nlp_scaling_method", "none") #set none scaling method
            set_optimizer_attribute(model, "tol",  tol) #High accuracy
            if cbs !== nothing
                cb_Ipopt = (args...) -> Ipopt_callback_iter(args..., cbs)
                MOI.set(model, Ipopt.CallbackFunction(), cb_Ipopt) #set termination depends only on Ipopt_callback
            end
        end

    elseif str == "madnlp" #for NLP
        model = Model(()->MadNLP.Optimizer())
        MOI.set(model, MOI.Silent(), false)
        set_optimizer_attribute(model, "print_level", 5) 

        set_optimizer_attribute(model, "blas_num_threads", nthreads())      
        set_optimizer_attribute(model, "tol", tol) 
        set_optimizer_attribute(model, "nlp_scaling", false)
    elseif str == "gurobi"
        model = Model(Gurobi.Optimizer)
    elseif str =="mosek"
        model = Model(Mosek.Optimizer)
    elseif str == "osqp"   #for LP, QP 
        model = Model(OSQP.Optimizer); 
        set_optimizer_attribute(model, "eps_abs", tol); 
        set_optimizer_attribute(model, "eps_rel", tol)

    elseif str == "clarabel" #for NLP
        model = Model(Clarabel.Optimizer)

    elseif str == "ecos" #for NLP
        model = Model(ECOS.Optimizer)

    else
        error("Specified solver is not supported")
    end

    set_silent(model)

    return model
end  
