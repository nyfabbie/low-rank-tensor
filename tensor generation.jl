using LinearAlgebra, Random


#generae 3 way tensors in Kruskal structure by rank and specified random generator
function generate_tensor(dims::Tuple{Int,Int,Int}, rank::Int; snr_db = Inf, collinearity = 0.0, weight = ones(Float64, rank), seed = 1, reseed = true)
    
    if reseed
        Random.seed!(seed)
    end
    m, n, p = dims

    A = randn(m, rank)
    B = randn(n, rank)
    C = randn(p, rank)
    return kruskal_3way_tensor(m, n, p, rank, weight, A, B, C)
end

#generate d-way tensors in Kruskal structure by rank and specified random generator
function generate_tensor(d::Int, dims::Array{Int}, rank::Int; weight = ones(Float64, rank), seed = 1, reseed = true)
    @assert length(dims) == d
    if reseed
        Random.seed!(seed)
    end
    factors = Vector{Matrix{Float64}}(undef, d)
    for i in 1:d
        factors[i] = randn(dims[i], rank)
    end
    return kruskal_dway_tensor(dims, rank, weight, factors)
end