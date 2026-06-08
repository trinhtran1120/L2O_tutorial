function LME_ADMM(data::MPCData_eco, gradient::gradient_struct, aux_sol::Function)

    z      = zeros(Float64, dim, N)
    w      = zeros(Float64, dim, N)
    α      = zeros(Float64, dim, N)
    buffer = zeros(Float64, dim, N)

    n_mb = div(N-1, s_mb) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb + 1)


    let N   = data.N,
        dim = data.dim,
        ρ   = data.rho
        return @inbounds function solver(x0::Float64, load_fc::Vector{Float64}, gen_fc::Vector{Float64}, callback::Union{Function, Nothing} =  nothing; 
            tol::Float64 = 1e-4, max_iter::Int = 1000, verbose::Bool = false, γ = 1.2)

            fill!(w, 0.)
            fill!(α, 0.)

            total_time = 0.
            J = 0.

            for i in 1:max_iter
                # ==== z-update ====
                start_time = time()
                @. buffer = w + α
                z .= buffer .- mini_batch(local_gradients, buffer)./ρ
                @. z = γ * z + (1 - γ)* w

                total_time += time() - start_time

                # ==== w-update ====
                w[1,:], w[2,:], w[3,:], sol_time = aux_sol(z - α, x0, load_fc, gen_fc)
                total_time += sol_time

                ## ============== Calculate dual variables and check termination ===========
                start_time = time()
                @. buffer  = w - z
                @. α += buffer

                CALL_BACK_STATUS = false
                total_time += time() - start_time

                J = get_objective(data, w)

                if callback !== nothing
                    CALL_BACK_STATUS = callback(z, w, α, i, J, total_time)
                end

                TERMINATION_STATUS = CALL_BACK_STATUS || (maximum(buffer) < tol)

                if TERMINATION_STATUS
                    if verbose  println("Learning ADMM converges at iteration $i with objective value = $J")  end
                    break 
                end

                if i == max_iter  println("Can not find an accurate solution, returned a close feasibility solution.")  end
            end

            return w, total_time, J
        end
    end
end


function LME_ADMM_split(data::MPCData_eco, gradient::gradient_struct, aux_sol::Function)
    """
    ADMM with splitting the constraints into two update steps, one for dynamics and one for bounds.
    The v-update uses the dynamics constraints, while the w-update uses the bound constraints.
    The dynamics constraints are used for consensus: v = z and w = v.
    """

    z = zeros(Float64, data.dim+1, data.N)
    w = copy(z)
    v = copy(z)
    α = copy(z)
    β = copy(z)
    buffer1 = copy(z)
    buffer2 = copy(buffer1)

    n_mb = div(data.N - 1, s_mb) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb)


    let N   = data.N,
        dim = data.dim,
        ρ   = data.rho,
        u_min = data.u_min,
        u_max = data.u_max,
        x_min = data.x_min,
        x_max = data.x_max
        return @inbounds function solver(init::Float64, load_fc::Vector{Float64}, gen_fc::Vector{Float64}, callback = nothing; 
            tol::Float64 = 1e-4, max_iter::Int = 1000, verbose::Bool = false)

            fill!(z, 0.)
            fill!(v, 0.)
            fill!(β, 0.)

            J = 0
            start_time = time()

            for i in 1:max_iter
                # ==== z-update ====
                buffer1 .= v .+ β
                @views z[1:dim, :] .= buffer1[1:dim, :] .- mini_batch(local_gradients, buffer1[1:dim, :])./ρ
                @views copyto!(z[dim+1, :], buffer1[dim+1, :])  # no learning for state variable

                # ==== v-update ====
                # Use the equality constraints in v-update
                @. buffer1 = (z - β + w + α)/2
                v .= aux_sol(buffer1, init, load_fc, gen_fc)
                
                # ==== w-update ====
                # Use the inequality constraints in v-update
                @. w = v - α

                @views clamp!(w[2,:], u_min, u_max)
                @views clamp!(w[3,:], 0.,    Inf)
                @views clamp!(w[4,:], x_min, x_max)

                ## ============== Calculate dual variables and check termination ===========
                @. buffer1  = w - v
                @. buffer2  = v - z
                α .+= buffer1
                β .+= buffer2

                CALL_BACK_STATUS = false
                J = get_objective(data, v)

                if callback !== nothing
                    CALL_BACK_STATUS = callback(z, w, α, v, β, i, J)
                end

                TERMINATION_STATUS = CALL_BACK_STATUS || ((maximum(buffer1) < tol) && (maximum(buffer2) < tol))

                ## ============== Check termination ===========
                if TERMINATION_STATUS
                    if verbose
                        println("Learning ADMM spliting constraints (v-update) converges at iteration $i with objective value = $J")
                    end
                    break 
                end
            end

            return v, time() - start_time, J
        end
    end
end

@inbounds function get_objective(data::MPCData_eco, z::Matrix{Float64})
    J = sum(data.cost_func(z[1,k], z[2,k], z[3,k]) for k in 1:data.N)
    return J
end
