include("matrix_tools.jl")

function load_model(fname::String)
    data   = JSON3.read(read(fname, String))
    vecf64 = (Vector{Float64} ∘ vec)
    model = (
        U = convert_to_matrix.(data["U"]),
        W = convert_to_matrix.(data["W"]),
        a = vecf64(data["a"]), 
        b = vecf64.(data["b"]),
        c = Float64(data["c"]),
        v = vecf64(data["v"])
    )
    return Float64(data["rho"]), model
end

function convert_to_matrix(L)
    v = copy(hcat(L...)')
    isempty(v) ? Float64[] : v
end


struct ICNN_Layer
    U::Matrix{Float64}
    W::Matrix{Float64}
    b::Vector{Float64}
end

@inbounds function (m::ICNN_Layer)(x::Matrix{Float64}, z::Matrix{Float64})
    s = m.W * z + m.U * x .+ m.b
    return map(softplus, s), s  # convex & nondecreasing
end


struct ICNN
    U0::Matrix{Float64}
    b0::Vector{Float64}
    layers::Vector{ICNN_Layer}
    v::Vector{Float64}
    a::Vector{Float64}
    c::Float64
end

@inbounds function (m::ICNN)(x::Matrix{Float64})::Matrix{Float64}
    z = softplus.(m.U0 * x .+ m.b0)  # first layer (no state W)
    for layer in m.layers
        z, _ = layer(x, z)
    end
    f = @. m.v'*z + m.a'*x + m.c
    return f
end

mutable struct gradient_struct
    lenlay::Int
    m::ICNN
    s_store::NTuple
    σ_store::NTuple 
    z_store::NTuple

    init_grad_x::Matrix{Float64}
    init_dL_dz::Matrix{Float64}  
    grad_x_buf::Matrix{Float64} 
    dL_curr::Matrix{Float64}
    dL_next::Matrix{Float64}
end

function gradient_struct(m::ICNN, nbatch::Int, dim::Int)

    U0     = m.U0
    layers = m.layers
    lenlay = length(layers)

    layer_rows = hcat(size(U0, 1), [size(layer.W, 1) for layer in layers])
    store_len  = length(layer_rows)

    s_store     = ntuple(i -> zeros(Float64, layer_rows[i], nbatch), store_len)
    σ_store     = ntuple(i -> zeros(Float64, layer_rows[i], nbatch), store_len)
    z_store     = ntuple(i -> zeros(Float64, layer_rows[i], nbatch), store_len)
    init_grad_x = repeat(m.a, 1, nbatch)
    init_dL_dz  = repeat(m.v, 1, nbatch)

    grad_x_buf  = zeros(Float64, dim, nbatch)
    dL_curr     = zeros(Float64, size(m.v, 1), nbatch)
    dL_next     = zeros(Float64, size(m.v, 1), nbatch)

    return gradient_struct(lenlay, m, s_store, σ_store, z_store, init_grad_x, init_dL_dz, grad_x_buf, dL_curr, dL_next)
end
precompile(gradient_struct, (ICNN, Int, Int))


@inbounds function mini_batch(local_gradients::NTuple, batch::AbstractMatrix)
    data_size = size(batch, 2)
    n_mb      = div(data_size -1, s_mb) + 1
    out       = copy(batch)

    @threads for i in 1:n_mb
        if i == n_mb
            @views out[:, (data_size - s_mb + 1):data_size] .= local_gradients[i](batch[:, (data_size - s_mb + 1):data_size])
        else
            @views out[:,(i-1)*s_mb+1:i*s_mb] .= local_gradients[i](batch[:,(i-1)*s_mb+1:i*s_mb])
        end
    end
    return out
end


function (obj::gradient_struct)(x::AbstractMatrix{Float64})

    s_first = obj.s_store[1]
    nL = obj.lenlay

    mul!(s_first,      obj.m.U0, x)
    add_bias!(s_first, obj.m.b0)
    activation_sigma!(obj.z_store[1], obj.σ_store[1], s_first)

    for i in 1:nL
        layer  = obj.m.layers[i]
        s_next = obj.s_store[i+1]
        z_prev = obj.z_store[i]

        mul!(s_next, layer.W, z_prev)
        mmul_add_matrix!(s_next, layer.U, x)
        add_bias!(s_next, layer.b)
        activation_sigma!(obj.z_store[i+1], obj.σ_store[i+1], s_next)
    end

    copyto!(obj.dL_curr,    obj.init_dL_dz)
    copyto!(obj.grad_x_buf, obj.init_grad_x)

    for i in nL:-1:1
        layer = obj.m.layers[i]
        dL_ds = obj.σ_store[i+1]
        hadamard!(dL_ds, obj.dL_curr)
        mmul_add_matrix!(obj.grad_x_buf, layer.U', dL_ds)

        mul!(obj.dL_next, layer.W', dL_ds)
        obj.dL_curr, obj.dL_next = obj.dL_next, obj.dL_curr
    end

    dL_ds_first = obj.σ_store[1]
    hadamard!(dL_ds_first, obj.dL_curr)
    mmul_add_matrix!(obj.grad_x_buf, obj.m.U0', dL_ds_first)

    return obj.grad_x_buf
end

@inline function LU_decomp(x)
    return lu(x)
end
precompile(LU_decomp, (SparseMatrixCSC{Float64, Int64},))


function dynamics_projection(mpc_data::MPCData_eco)
    # An analytic solution for  min ||Qs - q||² s.t. Ms = b 
    # The following is for establishment of constraint Ms = b
    # Order of variables: s = [m, u, p, x]ᵀ  R^{4N)}

    A = mpc_data.A
    B = mpc_data.B
    N = mpc_data.N

    IN = Matrix{Float64}(I, N, N)

    Mu = vcat(B*IN, zeros(N)')
    Mx = zeros(N+1, N)
    Mx[N+1,N] =  1.
    Mx[1,1]   = -1.
    for i in 2:N
        Mx[i,i-1] = A
        Mx[i,i]   = -1.
    end

    M = hcat(zeros(N+1, N), Mu, zeros(N+1, N), Mx)
    M = vcat(M, hcat(IN, IN, -IN, zeros(N,N)))  
    Q = hcat(I(3N), zeros(3N,N))

    # Build KKT blocks
    Qs = sparse(Q)
    Ms = sparse(M)

    K = [2 * (Qs' * Qs)  Ms';
         Ms              spzeros(2N+1, 2N+1)]

    F = LU_decomp(K)

    RHS = MVector{6N+1}(zeros(Float64, 6N+1))

    let F  = F,
        Qs = Qs,
        N  = N,
        A  = A
        return @inbounds function proj(qm::Matrix{Float64}, init::Float64, load_fc::Vector{Float64}, gen_fc::Vector{Float64})
            q  = vec(qm')
            fill!(RHS, 0.)
            
            RHS[1:4N] .= 2.0 .* (Qs'*q[1:3N])
            RHS[4N+1] = -A*init
            RHS[5N+1] = init
            RHS[5N+2:6N+1] .= load_fc - gen_fc
            s = F \ RHS
            
            return reshape(s[1:4N], N, dim+1)'
        end
    end
end



function qp_project_dykstra(q::AbstractVector,
                            A::AbstractMatrix,
                            b::AbstractVector,
                            x_min::AbstractVector,
                            x_max::AbstractVector;
                            max_iter::Int = 5000,
                            tol::Real = 1e-8)

    n = length(q)
    m = size(A, 1)

    # precompute affine projection operator: P_A(x) = x - A' * (AA')⁻¹ (A*x - b)
    AA = A * A'
    F = cholesky(Symmetric(AA))  # assume full row rank
    function proj_affine!(y::AbstractVector, x::AbstractVector)
        tmp = A * x .- b
        tmp = F \ tmp
        y .= x .- A' * tmp
        return y
    end
    
    # box projection
    proj_box!(y, x) = (y .= clamp.(x, x_min, x_max))

    # initialize
    x = copy(q)
    proj_box!(x, x)              # start inside box
    r1 = zeros(eltype(q), n)
    r2 = zeros(eltype(q), n)
    y  = similar(q)
    z  = similar(q)

    for k in 1:max_iter
        x_prev = x

        # Step 1: project onto affine set with correction r1
        z .= x .+ r1
        proj_affine!(y, z)
        r1 .= z .- y

        # Step 2: project onto box with correction r2
        z .= y .+ r2
        proj_box!(x, z)
        r2 .= z .- x

        # convergence
        if norm(x - x_prev) <= tol * max(1.0, norm(x))
            eqres = (m > 0) ? norm(A * x - b) : 0.0
            return (x, k, eqres)
        end
    end

    eqres = (m > 0) ? norm(A * x - b) : 0.0
    return (x, max_iter, eqres)
end
