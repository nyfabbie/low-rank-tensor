include("tensor generation.jl")
include("tensor operation.jl")
using LinearAlgebra, Random, TensorToolbox, Plots, BenchmarkTools, StatsPlots, DataFrames

#functions for CP-ALS. Takes in the tensor array, tensor rank, tolerance, iteration cap, random generator, initial factor matrices and return the resulting factor matrices as Kruskal tensors, history of relative error over iterations, whether the approximation reached convergence within the iteration cap, and the number of iteration required to reach convergence


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

function cp_als_dway(X, r::Int; tolerance::Float64 = 1e-8, maxiters::Int = 7000, init = :rand, seed::Int = 1, dimorder = nothing)
    dims = collect(size(X))
    d = length(dims)

    order = (dimorder === nothing) ? collect(1:d) : dimorder
    @assert length(order) == d
    convergence = true
    iteration = 0
    χ2 = sum(abs2, X)
    χ = sqrt(χ2)

    loss_hist = Float64[]

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

    et_prev = -Inf

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
        et = sqrt(res2) / χ
        fit = (χ == 0.0) ? 1.0 : (1.0 - sqrt(res2) / χ)
        push!(fit_hist, et)



        if iter > 1 && abs(et - et_prev) < tolerance
            #println("CP-ALS: Convergence achieved after $iter iterations with fit: $fit")
            convergence = true
            iteration = iter
            break
        end
        et_prev = et
        if iter == maxiters
            println("CP-ALS: Maximum iterations reached without convergence. Final fit: $fit")
            convergence = false
        end
    end
    return kruskal_dway_tensor(dims, r, λ, A), fit_hist, convergence, iteration
end