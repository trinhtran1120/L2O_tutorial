# Problem setup for the energy-management MPC example.
using Ipopt  
using OSQP   
using HiGHS
using SCS
using MadNLP
using ParameterJuMP
using JuMP
using Clarabel
using ECOS
using CSV, DataFrames
using StaticArrays

struct eco_mpc
    """Stage cost parameters for the economic MPC objective.
    Fields:
    - r_ec: Energy-cost weight.
    - r_df: Discomfort penalty weight.
    - r_op: Operating penalty weight.
    - η: Battery efficiency.
    - dT: Time-step duration.
    - N: MPC horizon length.
    - a: Discomfort model parameter.
    """
    r_ec::Float64
    r_df::Float64
    r_op::Float64
    η::Float64
    dT::Float64
    N::Int
    a::Float64
end


function discomfort(p, a)
    """Evaluate the discomfort term for supplied power.
    Args:
        p: Supplied power.
        a: Discomfort model parameter.

    Returns:
        Scalar discomfort value.
    """
    return a/p - 1
end


function (obj::eco_mpc)(m, u, p, model = nothing)
    """Evaluate or model the stage cost.
    Args:
        m: Net grid exchange term.
        u: Battery control action.
        p: Supplied power entering the discomfort term.
        model: Optional JuMP model. When provided, epigraph variables and
            constraints are added for absolute-value and positive-part terms.

    Returns:
        Numeric cost when `model === nothing`, otherwise a JuMP expression.
    """
    r_ec  = obj.r_ec
    r_df  = obj.r_df
    r_op  = obj.r_op
    η     = obj.η
    dT    = obj.dT
    a = obj.a

    if model !== nothing
        su = @variable(model)
        sm = @variable(model)
        sd = @variable(model)

        @constraint(model,  u <= su)
        @constraint(model, -u <= su)

        @constraint(model,  sm >= 0)
        @constraint(model,  sm >= m)

        @constraint(model,  sd >= 0)
        @constraint(model,  sd >= discomfort(p, a))

        return r_ec*dT*(m + (1-η)/(2*sqrt(η))*su) + r_op*sm + r_df*sd
    else #just for computation
        return r_ec*dT*(m + (1-η)/(2*sqrt(η))*abs(u)) + r_op*max(m, 0) + r_df*max(discomfort(p, a), 0)
    end
end


struct MPCData_eco
    """Input data for the energy-management MPC problem.
    Fields:
    - A: State transition coefficient.
    - B: Control input coefficient.
    - r_ec: Energy-cost weight.
    - r_df: Discomfort penalty weight.
    - r_op: Operating penalty weight.
    - η: Battery efficiency.
    - BESS: Battery energy storage system capacity.
    - dT: Time-step duration.
    - a: Discomfort model parameter.
    - x_min: Minimum state of charge.
    - x_max: Maximum state of charge.
    - u_min: Minimum battery power.
    - u_max: Maximum battery power.
    - x0: Initial state of charge.
    - dim: Problem input dimension used by learned models.
    - N: MPC horizon length.
    - load_forecast: Load forecast time series.
    - gen_forecast: Generation forecast time series.
    - rho: ADMM penalty parameter.
    - cost_func: Stage cost object.
    """
    A::Float64
    B::Float64
    r_ec::Float64
    r_df::Float64
    r_op::Float64
    η::Float64
    BESS::Float64
    dT::Float64
    a::Float64
    x_min::Float64
    x_max::Float64
    u_min::Float64
    u_max::Float64
    x0::Float64
    dim::Int
    N::Int
    load_forecast::SVector
    gen_forecast ::SVector
    rho::Float64
    cost_func::eco_mpc
end

function energy_mag()
    """Build the default energy-management MPC instance.
    Returns:
        An instance of `MPCData_eco`.
    """
    dT    = 0.25
    A     = 1.
    BESS  = 500
    B     = - dT/BESS
    r_ec  = 0.1
    r_df  = 10.
    r_op  = 19.19
    eta   = 0.8
    a     = 50.
    x_min = 0.2
    x_max = 0.8
    u_max = 700
    u_min = -700
    x0    = 0.5
    dim   = 3
    N     = 96
    rho   = 1 

    # Forecast data are sampled at 15-minute intervals.
    path_power_gen_data  = joinpath(@__DIR__, "datasets", "PV_48h_15-min_150kW_San_Diego.csv")
    path_power_load_data = joinpath(@__DIR__, "datasets", "load_15min_max100kW_SanDiego_Building.csv")

    gf = CSV.read(path_power_gen_data, DataFrame)[:,2]
    gen_forecast = SVector{size(gf,1)}(gf)
    lf = CSV.read(path_power_load_data, DataFrame)[:,2]
    load_forecast = SVector{size(lf,1)}(lf) 

    cost_func = eco_mpc(r_ec, r_df, r_op, eta, dT, N, a)

    return MPCData_eco(A, B, r_ec, r_df, r_op, eta, BESS, dT, a, x_min, x_max, u_min, u_max, x0, dim, N, load_forecast, gen_forecast, rho, cost_func)
end
