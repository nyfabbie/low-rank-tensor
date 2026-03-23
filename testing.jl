include("low rank tensor decomposition.jl")
using BenchmarkTools, Plots, StatsPlots, Random, LinearAlgebra

#=
df = DataFrame(Dimension=Tuple{}[], time=Float64[])
Random.seed!(24)
als_times = Float64[]
als_memory = Int[]
als_allocations = Int[]
gradient_times = Float64[]
benchmark_times = Float64[]


for i in 3:150
    m, n, p = i, i, i
    A0, B0, C0 = randn(m, 1), randn(n, 1), randn(p, 1)
    v0 = mats2vec(A0, B0, C0)
    init = [A0, B0, C0]

    test = generate_3way_tensor((m, n, p), 1, seed=564)
    test_tensor = construct_kruskal(test)
    #cp_als_3way(test_tensor, 1, init=v0)
    #cp_opt_gradient_descent_3way(test_tensor, 1, init=v0)
    trial_als =  @benchmark cp_als_3way($test_tensor, 1, seed=18, init=$v0) samples=100
    push!(als_times, median(trial_als.times))

    push!(als_memory, trial_als.memory)
    push!(als_allocations, trial_als.allocs)
    
    println("Dimension: $i, ALS Time: $(median(trial_als.times)) ns, Memory: $(trial_als.memory) bytes, Allocations: $(trial_als.allocs)")

    #=
    trial_gradient = @btime cp_opt_gradient_descent_3way(test_tensor, 1, seed=18, init=v0) samples=1

    trial_benchmark = @btime TensorToolbox.cp_als(test_tensor, 1, init=init) samples=1

    =#
end
=#


#=
println("ALS: ")
display(trial_als)
=#

#=
println("Gradient Descent: ")
display(trial_gradient)

println("TensorToolbox: ")
display(trial_benchmark)
=#

#=
als_times_plot = plot(als_times, label="CP-ALS", xlabel="Tensor Dimension", ylabel="Time (ns)", title="Time vs Dimension Comparison")
savefig(als_times_plot, "als time wrt dimension plot.png")

als_memory_plot = plot(als_memory, label="CP-ALS", xlabel="Tensor Dimension", ylabel="Memory (bytes)", title="Memory Usage vs Dimension Comparison", yscale=:log10)
savefig(als_memory_plot, "als memory wrt dimension plot.png")

als_allocations_plot = plot(als_allocations, label="CP-ALS", xlabel="Tensor Dimension", ylabel="Allocations", title="Memory Allocations vs Dimension Comparison", yscale=:log10)
savefig(als_allocations_plot, "als allocations wrt dimension plot.png")
=#
#=
memory_plot = boxplot(["CP-ALS"], [als_memory], ylabel="Memory (bytes)", title="Memory Usage Comparison", legend=false, yscale=:log10, outliers=false)
boxplot!(memory_plot, ["Gradient Descent"], [gradient_memory], legend=false, outliers=false)
boxplot!(memory_plot, ["TensorToolbox"], [benchmark_memory], legend=false, outliers=false)
savefig(memory_plot, "memory box plot.png")
=#


#= Random.seed!(24)
A0, B0, C0 = randn(75, 1), randn(75, 1), randn(75, 1)
v0 = mats2vec(A0, B0, C0)
test = generate_3way_tensor((75, 75, 75), 1, seed=564)
test_tensor = construct_kruskal(test)
X, hist = cp_opt_gradient_descent_3way(test_tensor, 1, seed=18, init=v0)
print(hist)  =#

function test_for_als_size_limit(seed = 896451)
    Random.seed!(seed)
    dim = 1150
    while true
        test = generate_tensor(3, [dim, dim, dim], 1, seed=98654)
        A0, B0, C0 = randn(dim, 1), randn(dim, 1), randn(dim, 1)
        test = construct_kruskal(test)
        v = mats2vec(A0, B0, C0)
        try
            println("Testing CP-fg with tensor of dimension: $dim x $dim x $dim")
            @time cp_fg(test, v, [dim, dim, dim], 1, 3)
        catch e
            println("CP-fg failed for dimension: $dim x $dim x $dim with error: $e")
            break
        end
        dim += 5
    end



end

function box_plot_time_to_dims(rank = 1; seed=5325)
    
    dimensions = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    plot = boxplot(xlabel="Tensor Dimension: RxRxR", ylabel="Execution Time(ms)", title="CP-ALS Execution Time over Dimension", legend=false, yscale=:log10, outliers=false)
    initv = []
    for i in dimensions
        test = generate_tensor(3, [i, i, i], rank, seed=6574)
        Random.seed!(seed)
        A0, B0, C0 = randn(i, 1), randn(i, 1), randn(i, 1)
        initv = [A0, B0, C0]
        test_full = construct_kruskal(test)
        benchmark = @benchmark cp_als_dway($test_full, 1, init=$initv) evals=1 samples=30 seconds=300
        plot = boxplot!(plot, [string(i)], benchmark.times .* 1e-6, legend=false, outliers=false)
        println("Completed benchmark for dimension: $i x $i x $i")
        println(benchmark.times)
    end
    savefig(plot, "cp_als_time_boxplot2.png")
end

box_plot_time_to_dims()




