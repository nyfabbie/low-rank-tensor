using LinearAlgebra, Random, TensorToolbox, Plots, BenchmarkTools, StatsPlots, DataFrames
include("tensor generation.jl")

# 3 way and d-way Kruskal tensor structure
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

#construct 3-way/d-way Kruskal tensors into full tensors
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

#stack factor matrices into one vector
function mats2vec(G1::Matrix{Float64}, G2::Matrix{Float64}, G3::Matrix{Float64})
    return vcat(vec(G1), vec(G2), vec(G3))
end

function mats2vec(G::Vector{Matrix{Float64}})
    return vcat((vec(M) for M in G)...)
end


#reverse of mats2vec
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

function khatri_rao(A::Matrix{Float64}, B::Matrix{Float64})
    p, r = size(A)
    n, _ = size(B)
    result = zeros(Float64, n*p, r)
    for j in 1:r
        result[:, j] = kron(A[:, j], B[:, j])
    end
    return result
end

#Khatri-Rao product computation
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

# Tensor times matrix mode 3 product
function ttm_mode3(X::Array{Float64, 3}, C::Matrix{Float64})
    m, n, p = size(X)
    s, _ = size(C)
    Xmat = reshape(X, m*n, p)
    Y = Xmat * C'
    return reshape(Y, m, n, s)
end

#Matrix times tensor Khatri Rao product
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

#normalize matrix columns
function columnnorm(A::Matrix{Float64}, r::Int)
    lambda = zeros(Float64, r)
    for j in 1:r
        lambda[j] = norm(A[:, j])
        A[:, j] /= lambda[j]
    end
    return A, lambda
end

#compute mode-k unfolding of tensors
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

#compute all mode-k unfoldings of tensors and return them in an array
function unfoldings(X)
    d = ndims(X)
    return [unfold_mode(X, k) for k in 1:d]
end