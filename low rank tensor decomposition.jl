using LinearAlgebra, Random, TensorToolbox, Plots, BenchmarkTools, StatsPlots

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


function generate_3way_tensor(dims, rank; snr_db = Inf, collinearity = 0.0, weight = ones(Float64, rank), seed = 1)
    #Todos: noise, collinearity, weight
    Random.seed!(seed)
    m, n, p = dims

    A = randn(m, rank)
    B = randn(n, rank)
    C = randn(p, rank)
    return kruskal_3way_tensor(m, n, p, rank, weight, A, B, C)
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


function vec2mats(v::Vector{Float64}, dims::Tuple{Int,Int,Int}, r::Int)
    m, n, p = dims
    A = reshape(v[1 : m*r],           m, r)
    B = reshape(v[m*r+1 : m*r+n*r],   n, r)
    C = reshape(v[m*r+n*r+1 : end],   p, r)
    return A, B, C
end

function mats2vec(G1::Matrix{Float64}, G2::Matrix{Float64}, G3::Matrix{Float64})
    return vcat(vec(G1), vec(G2), vec(G3))
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

function ttm_mode3(X::Array{Float64, 3}, C::Matrix{Float64})
    m, n, p = size(X)
    s, _ = size(C)
    Xmat = reshape(X, m*n, p)
    Y = Xmat * C'
    return reshape(Y, m, n, s)
end

function mttkrp_3way(X::Array{Float64, 3}, A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64}, mode::Int, dims::Tuple{Int,Int,Int}, r::Int)
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

function columnnorm(A::Matrix{Float64}, r::Int)
    lambda = zeros(Float64, r)
    for j in 1:r
        lambda[j] = norm(A[:, j])
        A[:, j] /= lambda[j]
    end
    return A, lambda
end

function cp_als_3way(X::Array{Float64, 3}, r::Int; tolerance::Float64 = 10^-8, max_iters::Int = 10000, seed::Int = 1, init::Vector{Float64} = nothing)
    Random.seed!(seed)
    norm = LinearAlgebra.norm(X)
    m, n, p = size(X)

    if init === nothing
        A = zeros(m, r) 
        B = randn(n, r) # random initialization
        C = randn(p, r)
    else
        A, B, C = vec2mats(init, (m, n, p), r)
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
            break
        end
        prevt = e
    end
    
    return kruskal_3way_tensor(m, n, p, r, lambda, A, B, C)
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

function backtracking_line_search(fg, x, f, g, d; a0::Float64 = 1.0, τ::Float64 = 1e-4, shrink::Float64 = 0.5, max_ls::Int = 300000000)
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

function cp_opt_gradient_descent_3way(X::Array{Float64, 3}, r::Int; maxiters::Int = 10000, tolerance::Float64 = 1e-6, a0::Float64 = 1.0, τ::Float64 = 1e-4, shrink::Float64 = 0.5, max_ls::Int = 30, seed::Int = 1, init::Vector{Float64} = nothing)
    
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
            break
        end

        d = -g
        alpha, v_new, f_new, g_new, ls_iters = backtracking_line_search(fg, v, f, g, d; a0=a0, τ=τ, shrink=shrink, max_ls=max_ls)

        push!(alpha_old, alpha)
        push!(ls_old, ls_iters)

        v, f, g =  v_new, f_new, g_new
        push!(f_old, f)
        push!(g_old, norm(g))
        if k == maxiters
            println("Maximum iterations reached without convergence. Final gradient norm: $(norm(g)).")
        end
    end

    hist = (f = f_old, g = g_old, alpha = alpha_old, ls_iters = ls_old)
    return v, hist
end

Random.seed!(7)
m, n, p = 10, 10, 10
A0, B0, C0 = randn(m, 1), randn(n, 1), randn(p, 1)
v0 = mats2vec(A0, B0, C0)
init = [A0, B0, C0]

test = generate_3way_tensor((m, n, p), 1, seed=42)
test_tensor = construct_kruskal(test)

trial_als =  @benchmark cp_als_3way(test_tensor, 1, seed=18, init=v0)

trial_gradient = @benchmark cp_opt_gradient_descent_3way(test_tensor, 1, seed=18, init=v0)

trial_benchmark = @benchmark TensorToolbox.cp_als(test_tensor, 1, init=init)

println("ALS: ")
display(trial_als)

println("Gradient Descent: ")
display(trial_gradient)

println("TensorToolbox: ")
display(trial_benchmark)

times_als = collect(trial_als.times)

times_gradient = collect(trial_gradient.times)

times_benchmark = collect(trial_benchmark.times)

box_plot = boxplot(["CP-ALS"], [times_als], ylabel="Time (ns)", title="Execution Time Comparison", legend=false, yscale=:log10, outliers=false)
boxplot = boxplot!(box_plot, ["Gradient Descent"], [times_gradient], legend=false, outliers=false)
boxplot = boxplot!(box_plot, ["TensorToolbox"], [times_benchmark], legend=false, outliers=false)
savefig(box_plot, "box.png")

