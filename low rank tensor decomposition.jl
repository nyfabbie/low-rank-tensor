using LinearAlgebra, Random, TensorToolbox, Plots, BenchmarkTools, StatsPlots, DataFrames

struct kruskal_3way_tensor
  m::Int
  n::Int
  p::Int
  r::Int
  weight::Vector{Float64}
  A::Matrix{Float64}
  B::Matrix{Float64}
  C::Matrix{Float64}
end

struct kruskal_dway_tensor
    dims::Vector{Int}
    r::Int
    weight::Vector{Float64}
    factors::Vector{Matrix{Float64}}
end



function generate_tensor(dims::Tuple{Int,Int,Int}, rank::Int; snr_db = Inf, collinearity = 0.0, weight = ones(Float64, rank), seed = 1)
    #Todos: noise, collinearity, weight
    Random.seed!(seed)
    m, n, p = dims

    A = randn(m, rank)
    B = randn(n, rank)
    C = randn(p, rank)
    return kruskal_3way_tensor(m, n, p, rank, weight, A, B, C)
end

function generate_tensor(d::Int, dims::Array{Int}, rank::Int; weight = ones(Float64, rank), seed = 1)
    @assert length(dims) == d
    Random.seed!(seed)
    factors = Vector{Matrix{Float64}}(undef, d)
    for i in 1:d
        factors[i] = randn(dims[i], rank)
    end
    return kruskal_dway_tensor(dims, rank, weight, factors)
end

function construct_kruskal(X::kruskal_3way_tensor, implementation = :efficient)
    m, n, p, r = X.m, X.n, X.p, X.r
    if implementation == :naive
        L = khatri_rao(X.C * Diagonal(X.weight), X.B, X.A)
        v = L * ones(Float64, r)
        return reshape(v, m, n, p)
    elseif implementation == :efficient
        R = khatri_rao(X.C, X.B)
        Y = (X.A * Diagonal(X.weight)) * R'
        return reshape(Y, m, n, p)
    end
end

function construct_kruskal(X::kruskal_dway_tensor)
    d = length(X.dims)
    @assert d >= 3
    k = fld(d, 2)
    L = khatri_rao(collect(X.factors[i] for i in k:-1:1))
    R = khatri_rao(collect(X.factors[i] for i in d:-1:k+1))
    Y = L * Diagonal(X.weight) * R'
    return reshape(Y, X.dims...)
end

function vec2mats(v::Vector{Float64}, dims::Tuple{Int,Int,Int}, r::Int)
    m, n, p = dims
    A = reshape(v[1 : m*r],           m, r)
    B = reshape(v[m*r+1 : m*r+n*r],   n, r)
    C = reshape(v[m*r+n*r+1 : end],   p, r)
    return A, B, C
end

function vec2mats(v::Vector{Float64}, dims::Array{Int}, r::Int)
    d = length(dims)
    A = Vector{Matrix{Float64}}(undef, d)
    idx = 1
    for k in 1:d
        nk = dims[k]
        A[k] = reshape(v[idx : idx + nk * r - 1], nk, r)
        idx += nk * r
    end
    return A
end

function mats2vec(G1::Matrix{Float64}, G2::Matrix{Float64}, G3::Matrix{Float64})
    return vcat(vec(G1), vec(G2), vec(G3))
end

function mats2vec(G::Vector{Matrix{Float64}})
    return vcat((vec(M) for M in G)...)
end

function khatri_rao(A::Matrix{Float64}, B::Matrix{Float64})
    p, r = size(A)
    n, _ = size(B)
    result = zeros(Float64, n*p, r)
    for j in 1:r
        result[:, j] = kron(A[:, j], B[:, j])
    end
    return result
end

function khatri_rao(A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64})
    m, r = size(A)
    n, _ = size(B)
    p, _ = size(C)
    result = zeros(Float64, m * n * p, r)
    for j in 1:r
        result[:, j] = kron(kron(A[:, j], B[:, j]), C[:, j])
    end
    return result
end

function khatri_rao(mats::Vector{Matrix{Float64}})
    @assert !isempty(mats)
    K = mats[1]
    for i in 2:length(mats)
        K = khatri_rao(K, mats[i])
    end
    return K
end

function ttm_mode3(X::Array{Float64, 3}, C::Matrix{Float64})
    m, n, p = size(X)
    s, _ = size(C)
    Xmat = reshape(X, m*n, p)
    Y = Xmat * C'
    return reshape(Y, m, n, s)
end

function mttkrp_3way(X, A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64}, mode::Int, dims::Tuple{Int,Int,Int}, r::Int)
    m, n, p = dims
    if mode == 1
        K = khatri_rao(C, B)
        Xmat = reshape(X, m, (n * p))
        return Xmat * K
    elseif mode == 2
        Y = ttm_mode3(X, copy(C'))
        V = zeros(Float64, n, r)
        @inbounds for l in 1:r
            V[:, l] .= (@view Y[:, :, l])' * (@view A[:, l])
        end
        return V
    elseif mode == 3
        K = khatri_rao(B, A)
        Xmat = reshape(X, (m * n), p)
        return Xmat' * K
    end
end

function mttkrp_dway(unfoldings, A::Vector{Matrix{Float64}}, k::Int)
    d = length(A)
    idxs = [j for j in 1:d if j != k]
    KR = khatri_rao([A[j] for j in reverse(idxs)])
    return unfoldings[k] * KR
end

function columnnorm(A::Matrix{Float64}, r::Int)
    lambda = zeros(Float64, r)
    for j in 1:r
        lambda[j] = norm(A[:, j])
        A[:, j] /= lambda[j]
    end
    return A, lambda
end

function unfold_mode(X, k::Int)
    d = ndims(X)
    dims = size(X)
    @assert 1 <= k <= d
    perm = vcat(k, collect(1:d)[collect(1:d) .!= k])
    Xp = permutedims(X, perm)
    nk = dims[k]
    rest = Int(prod(dims) / nk)
    return reshape(Xp, nk, rest)
end

function unfoldings(X)
    d = ndims(X)
    return [unfold_mode(X, k) for k in 1:d]
end

function cp_als_3way(X::Array{Float64, 3}, r::Int; tolerance::Float64 = 10^-8, max_iters::Int = 10000, seed::Int = 1, init = :rand)
    Random.seed!(seed)
    norm = LinearAlgebra.norm(X)
    m, n, p = size(X)

    if init == :rand
        A = zeros(m, r) 
        B = randn(n, r) # random initialization
        C = randn(p, r)
    else
        A, B, C = init
    end

    S2 = B' * B
    S3 = C' * C
    prevt = 0.0
    lambda = zeros(Float64, r)
    for iter in 1:max_iters
        U1 = mttkrp_3way(X, A, B, C, 1, (m, n, p), r)
        V1 = S3 .* S2
        A = U1 * inv(V1)
        A, _ = columnnorm(A, r)
        S1 = A' * A

        U2 = mttkrp_3way(X, A, B, C, 2, (m, n, p), r)
        V2 = S3 .* S1
        B = U2 * inv(V2)
        B, _ = columnnorm(B, r)
        S2 = B' * B 

        U3 = mttkrp_3way(X, A, B, C, 3, (m, n, p), r)
        V3 = S2 .* S1
        C = U3 * inv(V3)
        C, lambda = columnnorm(C, r)
        S3 = C' * C
        
        d = vec(sum(C .* U3, dims=1))
        alpha = dot(d, lambda)
        beta = dot(lambda, (V3 .* S3) * lambda)
        e = sqrt(abs(norm^2 - 2 * alpha + beta))
        if (iter > 1) && (abs(e - prevt) < tolerance * norm)
            #println("ALS: Dimension: $m x $n x $p, Convergence achieved after $iter iterations with error: $e")
            break
        end
        prevt = e
        if iter == max_iters
            #println("ALS: Dimension: $m x $n x $p, Maximum iterations reached without convergence. Final error: $e")
        end
    end
    
    return kruskal_3way_tensor(m, n, p, r, lambda, A, B, C)
end

function cp_als_dway(X, r::Int; tolerance::Float64 = 1e-8, maxiters::Int = 200, init = :rand, seed::Int = 1, dimorder = nothing)
    dims = collect(size(X))
    d = length(dims)

    order = (dimorder === nothing) ? collect(1:d) : dimorder
    @assert length(order) == d

    χ2 = sum(abs2, X)
    χ = sqrt(χ2)

    A = Vector{Matrix{Float64}}(undef, d)
    if init == :rand
        Random.seed!(seed)
        for i in 1:d
            A[i] = randn(dims[i], r)
        end
    elseif init isa Vector{Matrix{Float64}}
        @assert length(init) == d
        for k in 1:d
            @assert size(init[k]) == (dims[k], r)
            A[k] = copy(init[k])
        end
    else
        error("Invalid initialization method. Use :rand or provide a vector of matrices.")
    end

    λ = ones(Float64, r)
    for k in 1:d
        A[k], λk = columnnorm(A[k], r)
        λ .*= λk
    end

    S = [A[k]' * A[k] for k in 1:d]

    fit_hist = Float64[]
    fit_prev = -Inf

    unfoldings_X = unfoldings(X)

    for iter in 1:maxiters
        for k in order
            Uk = mttkrp_dway(unfoldings_X, A, k)
            Vk = ones(Float64, r, r)
            @inbounds for j in 1:d
                if j != k
                    Vk .*= S[j]
                end
            end
            A[k] = Uk / Vk
            A[k], λk = columnnorm(A[k], r)
            if k == order[end]
                λ = λk
            end
            S[k] = A[k]' * A[k]
        end
        Ud = mttkrp_dway(unfoldings_X, A, d)
        dcol = vec(sum(A[d] .* Ud, dims=1))
        alpha = dot(dcol, λ)

        H = ones(Float64, r, r)
        @inbounds for j in 1:d
            H .*= S[j]
        end
        beta = dot(λ, H * λ)

        res2 = max(χ2 - 2 * alpha + beta, 0.0)
        fit = (χ == 0.0) ? 1.0 : (1.0 - sqrt(res2) / χ)
        push!(fit_hist, fit)

        if iter > 1 && abs(fit - fit_prev) < tolerance
            #println("CP-ALS: Convergence achieved after $iter iterations with fit: $fit")
            break
        end
        fit_prev = fit
        if iter == maxiters
            println("CP-ALS: Maximum iterations reached without convergence. Final fit: $fit")
        end
    end
    return kruskal_dway_tensor(dims, r, λ, A), fit_hist
end

function cp_fg(X::Array{Float64,3}, v::Vector{Float64}, dims::Tuple{Int,Int,Int}, r::Int)
    A, B, C = vec2mats(v, dims, r)

    S1 = A' * A   
    S2 = B' * B   
    S3 = C' * C  


    G1 = A * (S3 .* S2) - mttkrp_3way(X, A, B, C, 1, dims, r) 
    G2 = B * (S3 .* S1) - mttkrp_3way(X, A, B, C, 2, dims, r)   

    V3 = S2 .* S1                     
    U3 = mttkrp_3way(X, A, B, C, 3, dims, r)                     
    G3 = C * V3 - U3                  

    f = 0.5 * LinearAlgebra.norm(X)^2 - sum(C .* U3) + 0.5 * sum((C' * C) .* V3)

    g = mats2vec(G1, G2, G3)

    return f, g
end

function cp_fg(X, v::Vector{Float64}, dims::Array{Int}, r::Int, order::Int)
    χ2 = sum(abs2, X)
    d = length(dims)
    @assert ndims(X) == d

    A = vec2mats(v, dims, r)
    S = [A[k]' * A[k] for k in 1:d]

    U = Vector{Matrix{Float64}}(undef, d)
    G = Vector{Matrix{Float64}}(undef, d)

    unfoldings_cpfg = unfoldings(X)

    for k in 1:d
        Uk = mttkrp_dway(unfoldings_cpfg, A, k)
        U[k] = Uk

        Vk = ones(Float64, r, r)
        @inbounds for j in 1:d
            if j != k
                Vk .*= S[j]
            end
        end
        G[k] = A[k] * Vk - Uk
    end

    inner_AU = sum(A[d] .* U[d])

    Vd = ones(Float64, r, r)
    @inbounds for j in 1:d-1
        Vd .*= S[j]
    end
    inner_VS = sum(Vd .* S[d])

    f = 0.5 * χ2 - inner_AU + 0.5 * inner_VS

    g = mats2vec(G)
    return f, g
end


function backtracking_line_search(fg, x, f, g, d; a0::Float64 = 1.0, τ::Float64 = 1e-4, shrink::Float64 = 0.5, max_ls::Int = 300)
    alpha = a0
    gd = dot(g, d)

    if gd >= 0
        d = -g
        gd = -dot(g, g)
    end

    for t in 1:max_ls
        x_new = x .+ alpha .* d
        f_new, g_new = fg(x_new)
        if f_new <= f + τ * alpha * gd
            return alpha, x_new, f_new, g_new, max_ls
        end
        alpha *= shrink
        if t == max_ls
        end
    end

    x_new = x .+ alpha .* d
    f_new, g_new = fg(x_new)
    return (alpha, x_new, f_new, g_new, max_ls)
end

function gradient_descent_3way(X::Array{Float64, 3}, r::Int; maxiters::Int = 7000, tolerance::Float64 = 1e-6, a0::Float64 = 1.0, τ::Float64 = 1e-4, shrink::Float64 = 0.5, max_ls::Int = 30, seed::Int = 1, init::Vector{Float64} = nothing)
    
    dims = size(X)
    m, n, p = dims
    χ2 = sum(abs2, X)

    if init === nothing
        Random.seed!(seed)
        A0 = randn(m, r)
        B0 = randn(n, r)
        C0 = randn(p, r)
        v = mats2vec(A0, B0, C0)
    else
        v = init
    end

    fg = (vv) -> cp_fg(X, vv, dims, r)
    f, g = fg(v)

    f_old = Float64[f]
    g_old = Float64[norm(g)]
    alpha_old = Float64[]
    ls_old = Int[]

    for k in 1:maxiters
        gnorm = norm(g)
        if gnorm <= tolerance
            #println("Gradient Descent: Dimension: $m x $n x $p, Convergence achieved after $k iterations with gradient norm: $gnorm")
            break
        end

        d = -g
        println("conducting line search for iteration $k with gradient norm: $gnorm")
        alpha, v_new, f_new, g_new, ls_iters =  backtracking_line_search(fg, v, f, g, d; a0=a0, τ=τ, shrink=shrink, max_ls=max_ls)

        push!(alpha_old, alpha)
        push!(ls_old, ls_iters)

        v, f, g =  v_new, f_new, g_new
        push!(f_old, f)
        push!(g_old, norm(g))
        if k == maxiters
            #println("Gradient Descent: Dimension: $m x $n x $p, Maximum iterations reached without convergence. Final gradient norm: $(norm(g)).")
        end
    end

    hist = (f = f_old, g = g_old, alpha = alpha_old, ls_iters = ls_old)
    A, B, C = vec2mats(v, dims, r)
    
    return kruskal_3way_tensor(m, n, p, r, ones(Float64, 1), A, B, C), hist
end


function gradient_descent(X::Array{Float64}, r::Int; maxiters::Int = 7000, tolerance::Float64 = 1e-6, a0::Float64 = 1.0, τ::Float64 = 1e-4, shrink::Float64 = 0.5, max_ls::Int = 30, seed::Int = 1, init = :rand)
    dims = collect(size(X))
    d = length(dims)
    χ2 = sum(abs2, X)

    if init == :rand
        Random.seed!(seed)
        factors0 = [randn(dims[i], r) for k in 1:d]
        v = mats2vec(factors0)
    else
        v = copy(init)
    end

    fg = (vv) -> cp_fg(X, vv, dims, r, d)
    f, g = fg(v)

    f_old = Float64[f]
    g_old = Float64[norm(g)]
    alpha_old = Float64[]
    ls_old = Int[]

    for k in 1:maxiters
        gnorm = norm(g)
        if gnorm <= tolerance
            #println("Gradient Descent: Dimension: $(dims), Convergence achieved after $k iterations with gradient norm: $gnorm")
            break
        end

        dir = -g

        alpha, v_new, f_new, g_new, ls_iters = @time backtracking_line_search(fg, v, f, g, dir; a0=a0, τ=τ, shrink=shrink, max_ls=max_ls)
        push!(alpha_old, alpha)
        push!(ls_old, ls_iters)

        v, f, g = v_new, f_new, g_new
        push!(f_old, f)
        push!(g_old, norm(g))
    end
    hist = (f = f_old, g = g_old, alpha = alpha_old, ls_iters = ls_old)

    factors = vec2mats(v, dims, r)
    weight = ones(Float64, r)

    return kruskal_dway_tensor(dims, r, weight, factors), hist
end