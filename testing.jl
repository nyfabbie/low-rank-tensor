include("low rank tensor decomposition.jl")

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

test_dway = generate_tensor(3, [3, 3, 3], 1, seed=564)
test_3way = generate_tensor((3, 3, 3), 1, seed=564)
full_3way = construct_kruskal(test_3way)
full_dway= construct_kruskal(test_dway)
Random.seed!(24)
A, B, C = randn(3, 1), randn(3, 1), randn(3, 1)
v = mats2vec(A, B, C)
X3, hist3 = gradient_descent_3way(full_3way, 1, seed=18, init=v)
Xd, histd = gradient_descent(full_dway, 1, seed=18, init=v)
println(full_3way)



println(construct_kruskal(X3))

println(construct_kruskal(Xd))


