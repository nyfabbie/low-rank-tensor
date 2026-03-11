using LinearAlgebra, Random, TensorToolbox, Plots

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
            V .= (@view Y[:, :, l])' * (@view A[:, l])
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

function cp_als_3way(X::Array{Float64, 3}, r::Int, tolerance::Float64 = 10^-8, max_iters::Int = 1000)
    norm = LinearAlgebra.norm(X)
    m, n, p = size(X)
    A = zeros(m, r) 
    B = randn(n, r) # random initialization
    C = randn(p, r)
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
        e = sqrt(norm^2 - 2 * alpha + beta)
        if (iter > 1) && (e - prevt < tolerance * norm)
            println("Final iteration: $iter, error delta: $(e - prevt), stopping criteria: $(tolerance * norm)")
            break
        end
        prevt = e
    end
    
    return kruskal_3way_tensor(m, n, p, r, lambda, A, B, C)
end

function cp_fg(X::Array{Float64,3}, norm::Float64, v::Vector{Float64}, dims::Tuple{Int,Int,Int})
    m, n, p = dims

    A, B, C = vec2mats(v, dims, r)

    S1 = A' * A   # r × r
    S2 = B' * B   # r × r
    S3 = C' * C   # r × r

    # Khatri-Rao and Khatri-Rao-like products
    # Matricized tensor times Khatri-Rao product
    X1 = reshape(X, m, n*p)          # X_(1): mode-1 unfolding
    X2 = reshape(permutedims(X, (2,1,3)), n, m*p)  # X_(2): mode-2 unfolding
    X3 = reshape(permutedims(X, (3,1,2)), p, m*n)  # X_(3): mode-3 unfolding

    CoB = khatri_rao(C, B)   # (n*p) × r
    CoA = khatri_rao(C, A)   # (m*p) × r
    BoA = khatri_rao(B, A)   # (m*n) × r

    # Steps 5-7: Gradients G1, G2
    G1 = A * (S3 .* S2) - X1 * CoB   # ∂f/∂A
    G2 = B * (S3 .* S1) - X2 * CoA   # ∂f/∂B

    # Steps 8-10: Save intermediate results for G3 and f
    V3 = S2 .* S1                     # r × r, saving for reuse
    U3 = X3 * BoA                     # p × r, saving for reuse
    G3 = C * V3 - U3                  # ∂f/∂C

    # Step 11: function value
    f = 0.5 * χ - sum(C .* U3) + 0.5 * sum((C' * C) .* V3)

    # Step 12: pack gradients into a single vector
    g = mats2vec(G1, G2, G3)

    return f, g
end



"""function cp_function_gradient(X, v)
end"""

test = generate_3way_tensor((5, 10, 15), 2)
X = cp_als_3way(construct_kruskal(test), 1)
println("test: " * string(construct_kruskal(test)))
println("estimated: " * string(construct_kruskal(X)))