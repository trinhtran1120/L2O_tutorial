# JuMP-based baseline solver and benchmark helper for the energy-management MPC example.

function mpc_eco_solver(name, mpc_para, tol, cbs::Union{Nothing, callback_struct} = nothing, init_val::Union{Matrix{Float64}, Nothing} = nothing)
    """Build a reusable JuMP solver for the energy-management MPC problem.
    Args:
        name: Solver name accepted by `pick_solver`.
        mpc_para: Energy-management MPC data containing dynamics, bounds, forecasts, and cost.
        tol: Solver tolerance passed to `pick_solver`.
        cbs: Optional callback state for solvers that support callback logging.
        init_val: Optional warm-start values for m, u, and p.

    Returns:
        A solver function that accepts an initial state, load forecast, and generation forecast.
    """
    model = pick_solver(name, tol, cbs)

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

    cost_func = mpc_para.cost_func

    @variable(model, m[1:N])
    @variable(model, u_min  .<= u[1:N] .<= u_max)
    @variable(model, p[1:N] .>= 0)

    if init_val !== nothing
        JuMP.set_start_value.(m, init_val[1,:])
        JuMP.set_start_value.(u, init_val[2,:])
        JuMP.set_start_value.(p, init_val[3,:])
    end

    @variable(model, x_min  .<= x[0:N] .<= x_max)

    @variable(model, para[1:3, 1:N] in MOI.Parameter.(ones(3, N)))
    @variable(model, x0             in MOI.Parameter.(x0_init))
    @variable(model, load[1:N]      in MOI.Parameter.(load_fc[1:N]))
    @variable(model, generator[1:N] in MOI.Parameter.(gen_fc[1:N]))

    # Battery state-of-charge dynamics over the MPC horizon.
    for i in 0:N-1
        @constraint(model, x[i+1] == x[i] -  dT*u[i+1]/BESS)
    end

    @constraint(model, x[N] == x0)
    @constraint(model, x[0] == x0)

    # Power balance: battery action, grid exchange, generation, load, and supplied power.
    @constraint(model, u + m + generator - load - p .== 0)

    J = sum(cost_func(m[k], u[k], p[k], model) for k in 1:N)
    @objective(model, Min, J)
    optimize!(model)

    if cbs !== nothing
        cbs.n_iter = Int[]
        cbs.rel_opt_gap = Float64[]
    end

    function solver(init::Float64, load::Vector{Float64}, generator::Vector{Float64}; verbose = false)
        """Solve the MPC problem for a new initial state and forecast.
        Args:
            init: Initial state of charge.
            load: Load forecast over the horizon.
            generator: Generation forecast over the horizon.
            verbose: Prints objective and iteration diagnostics when enabled.

        Returns:
            vars: Matrix whose rows contain m, u, and p trajectories.
            solve_time: Solver-reported solve time.
            objective_value: Objective value at the solution.
        """
        JuMP.set_parameter_value.(model[:load], load)
        JuMP.set_parameter_value.(model[:generator], generator)
        JuMP.set_parameter_value(model[:x0], init)

        optimize!(model)

        if verbose
            solver_name = lowercase(solution_summary(model).solver)
            J = objective_value(model)
            if solver_name == "ipopt"
                iter_count = barrier_iterations(model)
                println("Objective value = $J after $iter_count iterations")
            elseif solver_name == "madnlp"
                iter_count = barrier_iterations(model)
                println("Objective value = $J after $iter_count iterations")
            end
        end

        vars = vcat(JuMP.value.(model[:m])',  JuMP.value.(model[:u])', JuMP.value.(model[:p])')

        return vars, solve_time(model), objective_value(model)
    end

    return solver
end


function get_benchmark(func_return_time, args::Tuple, n_sample::Int = 100)
    """Fit a normal distribution to solver times, excluding the first timing sample.
    Args:
        func_return_time: Function returning `(solution, solve_time, objective_value)`.
        args: Arguments passed to `func_return_time`.
        n_sample: Number of timing samples to collect.

    Returns:
        A fitted Normal distribution of solver times after dropping the first sample.
        J: Objective value from the last solver run.
    """
    time_arr = zeros(n_sample)
    J = 0.

    for i in 1:n_sample
        _, time_arr[i], J = func_return_time(args...)
        if i % 10 == 0 GC.gc() end
    end

    return fit(Normal, time_arr[2:end]), J
end
