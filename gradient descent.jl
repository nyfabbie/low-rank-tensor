using LinearAlgebra, Random, TensorToolbox, Plots, BenchmarkTools, StatsPlots, DataFrames
include("tensor generation.jl")
include("tensor operation.jl")


#computation of CP objective function. Takes in the tensor array, tensor norm, all mode-k unfoldings of the tensor, mats2vec of factor matrices, dimension of the tensor, tensor rank and tensor order, and return the objective function as a float
function cp_function(X, χ2, unfoldings_cpf, v::Vector{Float64}, dims::Array{Int}, r::Int, order::Int)
    A = vec2mats(v, dims, r)
    Ad = A[order]
    S = [A[k]' * A[k] for k in 1:order]
    Sd = S[order]
    Ud = mttkrp_dway(unfoldings_cpf, A, order)
    Vd = ones(Float64, r, r)
    @inbounds for j in 1:order-1
        Vd .*= S[j]
    end

    inner_AU = sum(Ad .* Ud)
    inner_VS = sum(Vd .* Sd)
    return 1/2 * χ2 - inner_AU + 1/2 * inner_VS
end

#compute the CP objective function and gradient
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



function cp_fg(X, χ2, unfoldings_cpfg, v::Vector{Float64}, dims::Array{Int}, r::Int, order::Int)

    A = vec2mats(v, dims, r)
    S = [A[k]' * A[k] for k in 1:order]

    U = Vector{Matrix{Float64}}(undef, order)
    G = Vector{Matrix{Float64}}(undef, order)


    for k in 1:order
        Uk = mttkrp_dway(unfoldings_cpfg, A, k)
        U[k] = Uk

        Vk = ones(Float64, r, r)
        @inbounds for j in 1:order
            if j != k
                Vk .*= S[j]
            end
        end
        G[k] = A[k] * Vk - Uk
    end

    inner_AU = sum(A[order] .* U[order])

    Vd = ones(Float64, r, r)
    @inbounds for j in 1:order-1
        Vd .*= S[j]
    end
    inner_VS = sum(Vd .* S[order])

    f = 0.5 * χ2 - inner_AU + 0.5 * inner_VS

    g = mats2vec(G)
    return f, g
end

#conducting backtracking line search, return the step size and number of iterations
function backtracking_line_search(f_only, x, f, g, d; a0::Float64 = 1.0, τ::Float64 = 1e-4, shrink::Float64 = 0.5, max_ls::Int = 30)
    alpha = a0
    gd = dot(g, d)
    if gd >= 0
        d = -g
        gd = -dot(g, g)
    end
    for t in 1:max_ls
        f_new = f_only(x .+ (alpha .* d))
        if f_new <= f + τ * alpha * gd
            #println("Line Search: Sufficient decrease achieved after $t iterations with step size: $alpha")
            return alpha, t
        end
        alpha *= shrink
    end
    return alpha, max_ls
end


#performing gradient descnet. return the resulting factor matrices as Kruskal structure, history of gradient norm over iterations, whether the approximation reched convergence within the iteration cap, and the number of iterations
function gradient_descent_3way(X::Array{Float64, 3}, r::Int; maxiters::Int = 20000, tolerance::Float64 = 1e-6, a0::Float64 = 1.0, τ::Float64 = 1e-4, shrink::Float64 = 0.5, max_ls::Int = 30, seed::Int = 1, init::Vector{Float64} = nothing)
    
    dims = size(X)
    m, n, p = dims
    χ2 = sum(abs2, X)
    convergence = true
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

    loss_hist = Float64[]

    for k in 1:maxiters
        gnorm = norm(g)
        if gnorm <= tolerance
            #println("Gradient Descent: Dimension: $m x $n x $p, Convergence achieved after $k iterations with gradient norm: $gnorm")
            convergence = true
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
            convergence = false
        end
    end

    hist = (f = f_old, g = g_old, alpha = alpha_old, ls_iters = ls_old)
    A, B, C = vec2mats(v, dims, r)
    
    return kruskal_3way_tensor(m, n, p, r, ones(Float64, 1), A, B, C), hist, convergence
end




function gradient_descent(X::Array{Float64}, r::Int; maxiters::Int = 7000, tolerance::Float64 = 1e-6, a0::Float64 = 1.0, τ::Float64 = 1e-4, shrink::Float64 = 0.5, max_ls::Int = 30, seed::Int = 1, init = :rand)
    dims = collect(size(X))
    d = length(dims)
    χ2 = sum(abs2, X)
    unfoldings_cpfg = unfoldings(X)
    iteration = maxiters
    convergence = false
    if init == :rand
        Random.seed!(seed)
        factors0 = [randn(dims[i], r) for k in 1:d]
        v = mats2vec(factors0)
    else
        v = copy(init)
    end

    fg = (vv) -> cp_fg(X, χ2, unfoldings_cpfg, vv, dims, r, d)
    f_only = (vv) -> cp_function(X, χ2, unfoldings_cpfg, vv, dims, r, d)
    f, g = fg(v)

    f_old = Float64[f]
    g_old = Float64[norm(g)]
    alpha_old = Float64[]
    ls_old = Int[]


    for k in 1:maxiters
        gnorm = norm(g) 
        if gnorm <= tolerance
            #println("Gradient Descent: Dimension: $(dims), Convergence achieved after $k iterations with gradient norm: $gnorm")
            convergence = true
            iteration = k
            break
        end

        dir = -g

        alpha, ls_iters = backtracking_line_search(f_only, v, f, g, dir; a0=a0, τ=τ, shrink=shrink, max_ls=max_ls)

        v_new = v .+ alpha .* dir
        f_new, g_new = fg(v_new)


        push!(alpha_old, alpha)
        push!(ls_old, ls_iters)

        v, f, g = v_new, f_new, g_new
        push!(f_old, f)
        push!(g_old, norm(g))
    end
    hist = (f = f_old, g = g_old, alpha = alpha_old, ls_iters = ls_old)

    factors = vec2mats(v, dims, r)
    weight = ones(Float64, r)

    return kruskal_dway_tensor(dims, r, weight, factors), hist, convergence, iteration
end