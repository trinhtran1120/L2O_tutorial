# Conventional ADMM and proximal-data generation utilities for the energy-management MPC example.
using BlockArrays
using StaticArrays


mutable struct prime_sol_struct
    """Reusable primal-update model for economic MPC ADMM.
    Fields:
    - model: JuMP model for the proximal stage-cost update.
    - rho: ADMM penalty parameter.
    - dim: Number of decision variables per stage.
    - N: MPC horizon length.
    """
    model::Model
    rho::Float64
    dim::Int
    N::Int
end

function prime_sol_struct(name::String, mpc_para::MPCData_eco)
    """Build the horizon-wise primal proximal solver.
    Args:
        name: Solver name accepted by `pick_solver`.
        mpc_para: Energy-management MPC data.

    Returns:
        A callable `prime_sol_struct`.
    """
    model     = pick_solver(name)
    rho       = mpc_para.rho
    dim       = mpc_para.dim
    N         = mpc_para.N

    @variable(model, m[1:N])
    @variable(model, u[1:N])
    @variable(model, p[1:N] .>= 0)

    @variable(model, para[1:dim, 1:N] in MOI.Parameter.(ones(dim,N)))
    vars = vcat(m',u',p')

    J = sum(mpc_para.cost_func(m[i], u[i], p[i], model) + (rho/2)*(dot(vars[:,i], vars[:,i]) - 2*dot(para[:,i], vars[:,i])) for i in 1:N)

    @objective(model, Min, J)

    optimize!(model) #build model

    return prime_sol_struct(model, rho, dim, N)
end

function (obj::prime_sol_struct)(q::Matrix{Float64})
    """Solve the primal proximal update for a matrix of stage queries.
    Args:
        q: Query matrix whose columns correspond to MPC stages.

    Returns:
        vars: Matrix of m, u, and p primal updates.
        solve_time: Solver-reported solve time.
    """
    JuMP.set_parameter_value.(obj.model[:para], q)
    optimize!(obj.model)

    vars       = copy(q)
    vars[1,:] .= JuMP.value.(obj.model[:m]) 
    vars[2,:] .= JuMP.value.(obj.model[:u]) 
    vars[3,:] .= JuMP.value.(obj.model[:p])
    return vars, solve_time(obj.model)
end


function ADMM_eco_iter(data::MPCData_eco, prime_sol::prime_sol_struct, aux_sol::Function; max_iter = 1000, tol = tol)
    """Build the conventional ADMM solver for the energy-management MPC problem.
    Args:
        data: Energy-management MPC data.
        prime_sol: Primal proximal solver.
        aux_sol: Auxiliary projection solver.
        max_iter: Maximum ADMM iterations.
        tol: Residual tolerance used for termination.

    Returns:
        A solver function returning the primal matrix, total solve time, and objective value.
    """
    dim = data.dim
    N   = data.N
    z   = zeros(Float64, dim, N)
    z₊  = zeros(Float64, dim, N)

    w   = zeros(Float64, dim, N)
    α   = zeros(Float64, dim, N)
    buffer = zeros(Float64, dim, N)

    let N       = data.N, 
        x0      = data.x0, 
        load_fc = data.load_forecast[1:N], 
        gen_fc  = data.gen_forecast[1:N]
        return @inbounds function solver(init::Float64, load_fc::Vector{Float64}, gen_fc::Vector{Float64}, callback = nothing; verbose = false, γ = 1.2)        
            """Run conventional ADMM for one initial state and forecast."""

            fill!(w, 0.)
            fill!(α, 0.)

            total_time = 0.
            J = 0.
            for i in 1:max_iter
                # z-update: solve the proximal stage-cost problem.
                @. buffer = w + α

                z, sol_time = prime_sol(buffer)

                @. z = γ * z + (1 - γ) * w

                total_time += sol_time

                # w-update: project onto dynamics, bounds, and power-balance constraints.
                w[1,:], w[2,:], w[3,:], sol_time = aux_sol(z - α, init, load_fc, gen_fc)
                total_time += sol_time
                
                # Dual update and residual check.
                start_time = time()
                @. buffer  = w - z
                @. α += buffer

                CALL_BACK_STATUS = false
                J = get_objective(data, w)
                total_time += time() - start_time

                if callback !== nothing
                    CALL_BACK_STATUS = callback(z, w, α, i, J, total_time)
                end

                TERMINATION_STATUS = CALL_BACK_STATUS || (maximum(buffer) < tol)

                if TERMINATION_STATUS
                    if verbose
                        println("Conventional ADMM converges at iteration $i with objective value = $J")
                    end
                    break 
                end

                if i == max_iter println("Can not find an accurate solution, returned a close feasibility solution.") end
            end

            return w, total_time, J
        end
    end
end


function aux_solver_eco(solver_name::String, mpc_para::MPCData_eco)
    """Build the auxiliary projection solver used by conventional ADMM.
    Args:
        solver_name: Solver name accepted by `pick_solver`.
        mpc_para: Energy-management MPC data.

    Returns:
        A projection solver for a query matrix and current forecast.
    """
    model = pick_solver(solver_name)

    N    = mpc_para.N
    dim  = mpc_para.dim
    BESS = mpc_para.BESS
    dT   = mpc_para.dT

    u_max = mpc_para.u_max
    u_min = mpc_para.u_min

    x_max   = mpc_para.x_max
    x_min   = mpc_para.x_min
    x0_init = mpc_para.x0

    load_fc = mpc_para.load_forecast
    gen_fc  = mpc_para.gen_forecast

    @variable(model, m[1:N])
    @variable(model, u_min  .<= u[1:N] .<= u_max)
    @variable(model, p[1:N] .>= 1)
    @variable(model, x_min  .<= x[0:N] .<= x_max)

    @variable(model, para[1:dim,1:N] in MOI.Parameter.(ones(dim,N)))
    @variable(model, x0             in MOI.Parameter.(x0_init))
    @variable(model, load[1:N]      in MOI.Parameter.(load_fc[1:N]))
    @variable(model, generator[1:N] in MOI.Parameter.(gen_fc[1:N]))

    # Battery state-of-charge dynamics.
    for i in 0:N-1
        @constraint(model, x[i+1] == x[i] -  dT*u[i+1]/BESS)
    end

    @constraint(model, x[N] == x0)
    @constraint(model, x[0] == x0)

    # Power balance across each stage.
    @constraint(model, u + m + generator - load - p .== 0)

    vars = vcat(m', u', p')
    J    = sum(dot(vars[:,i], vars[:,i]) - 2*dot(para[:,i], vars[:,i]) for i in 1:N)
    @objective(model, Min, J)

    optimize!(model) #build model

    let model = model
        return function solver(q::Matrix{Float64}, init::Float64, load::Vector{Float64}, generator::Vector{Float64})
            JuMP.set_parameter_value.(model[:para], q)
            JuMP.set_parameter_value.(model[:load], load)
            JuMP.set_parameter_value.(model[:generator], generator)
            JuMP.set_parameter_value(model[:x0], init)

            optimize!(model)
            
            return JuMP.value.(model[:m]), JuMP.value.(model[:u]), JuMP.value.(model[:p]), solve_time(model)
        end
    end    
end



function ADMM_eco_iter_data(mpc_para::MPCData_eco, prime_sol::Function, aux_sol::Function; max_iter = 1000, tol = tol)
    """Build ADMM data-generation solver that records proximal samples.
    Args:
        mpc_para: Energy-management MPC data.
        prime_sol: Scalar primal proximal solver.
        aux_sol: Auxiliary projection solver.
        max_iter: Maximum ADMM iterations.
        tol: Residual tolerance used for termination.

    Returns:
        A solver function that appends proximal training data and returns the ADMM solution and objective.
    """
    N  = mpc_para.N
    ρ  = mpc_para.rho
    dim = mpc_para.dim
    cost_func = mpc_para.cost_func

    function solver(data, init = mpc_para.x0, load_fc = mpc_para.load_forecast[1:N], gen_fc = mpc_para.gen_forecast[1:N]; verbose = false)
        """Run ADMM and collect primal proximal training samples."""

        z = rand(dim, N)
        w = rand(dim, N)
        α = rand(dim, N)

        for i in 1:max_iter
            # z-update: solve one proximal subproblem per horizon stage.
            for k in 1:N
                q = w[:,k] + α[:,k]
                z[:,k], MEq = prime_sol(q)

                # Store query, Moreau envelope value, and proximal-gradient target.
                push!(data["input"], q)
                push!(data["env"],   MEq)
                push!(data["grad"],  ρ*(q - z[:,k]))
            end

            # w-update: project the shifted z variable onto MPC constraints.
            αᵣ = 1.1
            z₊ = αᵣ*z + (1 - αᵣ)*w
            w = aux_sol(z₊ - α, init, load_fc, gen_fc)

            # Dual update and residual check.
            α += w - z₊
            
            rmax = maximum(abs.(w-z))

            if rmax < tol
                if verbose == true
                    println("Conventional ADMM converges at iteration $i---tol=$tol")
                end
                break 
            end
        end

        J_ADMM = sum(cost_func(w[1,k], w[2,k], w[3,k]) for k in 1:N)

        return w, J_ADMM
    end

    return solver
end


function prime_solver_eco_data(name::String, mpc_para::MPCData_eco)
    """Build the scalar primal proximal solver used for data generation.
    Args:
        name: Solver name accepted by `pick_solver`.
        mpc_para: Energy-management MPC data.

    Returns:
        A solver for one stage query and its Moreau-envelope value.
    """
    model = pick_solver(name)
    ρ     = mpc_para.rho
    N     = mpc_para.N
    cost_func = mpc_para.cost_func

    @variable(model, m)
    @variable(model, u)
    @variable(model, p >= 0)

    @variable(model, para[1:3] in MOI.Parameter.(ones(3)))

    vars = [m, u, p] # Match the row order used by auxiliary solvers.

    J = cost_func(m, u, p, model) + (ρ/2)*(dot(vars, vars) - 2*dot(para, vars) + dot(para, para))
    @objective(model, Min, J)
    optimize!(model) #build model

    let model = model
        return @inbounds function solver(q::Vector{Float64})
            JuMP.set_parameter_value.(model[:para], q)
            optimize!(model)
            vars = [JuMP.value.(model[:m]), JuMP.value.(model[:u]), JuMP.value.(model[:p])]
            return vars, objective_value(model)
        end
    end

    return solver
end


function aux_solver_eco_data(solver_name::String, mpc_para::MPCData_eco)
    """Build the auxiliary projection solver used for data generation.
    Args:
        solver_name: Solver name accepted by `pick_solver`.
        mpc_para: Energy-management MPC data.

    Returns:
        A projection solver for a query matrix and current forecast.
    """
    model = pick_solver(solver_name)

    N    = mpc_para.N
    BESS = mpc_para.BESS
    dT   = mpc_para.dT

    u_max = mpc_para.u_max
    u_min = mpc_para.u_min

    x_max   = mpc_para.x_max
    x_min   = mpc_para.x_min
    x0_init = mpc_para.x0

    load_fc = mpc_para.load_forecast
    gen_fc  = mpc_para.gen_forecast

    @variable(model, m[1:N])
    @variable(model, u_min  .<= u[1:N] .<= u_max)
    @variable(model, p[1:N] .>= 0)
    @variable(model, x_min  .<= x[0:N] .<= x_max)

    @variable(model, para[1:3, 1:N] in MOI.Parameter.(ones(3, N)))
    @variable(model, x0             in MOI.Parameter.(x0_init))
    @variable(model, load[1:N]      in MOI.Parameter.(load_fc[1:N]))
    @variable(model, generator[1:N] in MOI.Parameter.(gen_fc[1:N]))

    # Battery state-of-charge dynamics.
    for i in 0:N-1
        @constraint(model, x[i+1] == x[i] -  dT*u[i+1]/BESS)
    end

    @constraint(model, x[N] == x0)
    @constraint(model, x[0] == x0)

    # Power balance across each stage.
    @constraint(model, u + m + generator - load - p .== 0)

    vars = vcat(m', u', p')
    J = sum(dot(vars[:,i], vars[:,i]) - 2*dot(para[:,i], vars[:,i]) for i in 1:N)
    @objective(model, Min, J)

    optimize!(model) #build model

    function solver(q::Matrix{Float64}, init::Float64, load::Vector{Float64}, generator::Vector{Float64})
        JuMP.set_parameter_value.(model[:para], q)
        JuMP.set_parameter_value.(model[:load], load)
        JuMP.set_parameter_value.(model[:generator], generator)
        JuMP.set_parameter_value(model[:x0], init)

        optimize!(model)
        
        vars = vcat(JuMP.value.(model[:m])',  JuMP.value.(model[:u])', JuMP.value.(model[:p])')

        return vars
    end

    return solver
end
