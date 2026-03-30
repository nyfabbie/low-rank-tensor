include("low rank tensor decomposition.jl")
using BenchmarkTools, Plots, StatsPlots, Random, LinearAlgebra, TensorToolbox

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

function als_box_plot_time_to_dims(rank = 1; seed=5325)
    
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

function gradient_box_plot_time_to_dims(rank = 1; seed=6619)
    dimensions = [10,20,30,40,50,60,70,80,90,100, 110, 120, 130, 140, 150]
    plot = plot(xlabel="Tensor Dimension: RxRxR", ylabel="Execution Time(ms)", title="CP-OPT Gradient Descent Execution Time over Dimension", legend=false, yscale=:log10, outliers=false)
    initv=[]
    for i in dimensions
        test = generate_tensor(3, [i, i, i], rank, seed=2747)
        Random.seed!(seed)
        A0, B0, C0 = randn(i, 1), randn(i, 1), randn(i, 1)
        initv = mats2vec(A0, B0, C0)
        test_full = construct_kruskal(test)
        benchmark = @benchmark gradient_descent($test_full, 1, init=$initv) evals=1 samples=20 seconds=5000
        plot = boxplot!(plot, [string(i)], benchmark.times .* 1e-6, legend=false, outliers=false)
        println("Completed benchmark for dimension: $i x $i x $i")
        println(benchmark.times)
    end
    savefig(plot, "cp_opt_gradient_descent_time_boxplot_10-150.png")
end

function als_plot_time_to_rank(seed=8834)
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plot = boxplot(xlabel="Tensor Rank", ylabel="Execution Time(ms)", title="CP-ALS Execution Time over Rank", legend=false, yscale=:log10, outliers=false)
    initv = []
    for r in ranks
        test = generate_tensor(3, [50, 50, 50], r, seed=311)
        Random.seed!(seed)
        A0, B0, C0 = randn(50, 1), randn(50, 1), randn(50, 1)
        initv = [A0, B0, C0]
        test_full = construct_kruskal(test)
        benchmark = @benchmark cp_als_dway($test_full, 1, init=$initv) evals=1 samples=25 seconds=5000
        plot = boxplot!(plot, [string(r)], benchmark.times .* 1e-6, legend=false, outliers=false)
        println("Completed benchmark for rank: $r")
        println(benchmark.times)
    end
    savefig(plot, "cp_als_time_boxplot_rank.png")
end

function gradient_descent_time_to_rank(seed=8834)
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plot = boxplot(xlabel="Tensor Rank", ylabel="Execution Time(ms)", title="Gradient Descent Execution Time over Rank", legend=false, yscale=:log10, outliers=false)
    initv = []
    for r in ranks
        test = generate_tensor(3, [50, 50, 50], r, seed=311)
        Random.seed!(seed)
        A0, B0, C0 = randn(50, 1), randn(50, 1), randn(50, 1)
        initv = mats2vec(A0, B0, C0)
        test_full = construct_kruskal(test)
        benchmark = @benchmark gradient_descent($test_full, 1, init=$initv) evals=1 samples=25 seconds=5000
        plot = boxplot!(plot, [string(r)], benchmark.times .* 1e-6, legend=false, outliers=false)
        println("Completed benchmark for rank: $r")
        println(benchmark.times)
    end
    savefig(plot, "gradient_descent_time_boxplot_rank.png")
end

function time_to_rank_comparison(; seed=4136, m=50, n=50, p=50, maxiters=5000, samples=50)
    ranks = collect(1:10)
    time_plot = boxplot(xlabel="Tensor Rank", ylabel="Execution Time(ms)", title="Execution Time over Rank Comparison", legend=:topleft, yscale=:log10, outliers=false)
    iteration_plot = boxplot(xlabel="Tensor Rank", ylabel="Iterations to Convergence", title="Iterations to Convergence over Rank Comparison", legend=:topleft, yscale=:log10, outliers=false)
    allocation_plot = boxplot(xlabel="Tensor Rank", ylabel="Memory Allocated (bytes)", title="Memory Allocated over Rank Comparison", legend=:topleft, yscale=:log10, outliers=false)
    allocation_per_iteration_plot = boxplot(xlabel="Tensor Rank", ylabel="Memory Allocated per Iteration (bytes)", title="Memory Allocated per Iteration over Rank Comparison", legend=:topleft, yscale=:log10, outliers=false)
    success_rate_plot = plot(xlabel="Tensor Rank", ylabel="Success Rate", title="Success Rate over Rank Comparison", legend=:topleft, yscale=:linear)
    time_per_iteration_plot = boxplot(xlabel="Tensor Rank", ylabel="Time per Iteration (ms)", title="Time per Iteration over Rank Comparison", legend=:topleft, yscale=:log10, outliers=false)
    initv = []
    first = true
    Random.seed!(seed)
    #= als_allocation_per_iteration_medians = Float64[]
    gradient_allocation_per_iteration_medians = Float64[] =#
    als_success_rates = Float64[]
    gradient_success_rates = Float64[]
    for r in ranks
        als_convergences = Bool[]
        gradient_convergences = Bool[]
        als_times = Float64[]
        gradient_times = Float64[]
        als_iterations = Int[]
        gradient_iterations = Int[]
        als_allocation_per_iteration = Float64[]
        gradient_allocation_per_iteration = Float64[]
        als_time_per_iteration = Float64[]
        gradient_time_per_iteration = Float64[]
        for i in 1:samples
            A0, B0, C0 = randn(m, 1), randn(n, 1), randn(p, 1)
            test = generate_tensor(3, [m, n, p], r, reseed=false)
            initv = mats2vec(A0, B0, C0)
            test_full = construct_kruskal(test)
            als_result, als_time, als_allocated, als_gc, als_memorycounters = @timed cp_als_dway(test_full, 1, init=[A0, B0, C0], maxiters=maxiters)
            gradient_result, gradient_time, gradient_allocated, gradient_gc, gradient_memorycounters = @timed gradient_descent(test_full, 1, init=initv, maxiters=maxiters)
            _, _, als_convergence, als_iteration, als_loss_hist = als_result
            _, _, gradient_convergence, gradient_iteration, gradient_loss_hist = gradient_result
            if als_convergence
                push!(als_times, als_time)
                push!(als_iterations, als_iteration)
                
            end
            if als_iteration > 0
                push!(als_allocation_per_iteration, als_allocated / als_iteration)
                push!(als_time_per_iteration, als_time / als_iteration)
            end
            if gradient_convergence
                push!(gradient_times, gradient_time)
                push!(gradient_iterations, gradient_iteration)
                
            end
            if gradient_iteration > 0
                push!(gradient_allocation_per_iteration, gradient_allocated / gradient_iteration)
                push!(gradient_time_per_iteration, gradient_time / gradient_iteration)
            end
            push!(als_convergences, als_convergence)
            push!(gradient_convergences, gradient_convergence)
        end
        if first
            boxplot!(time_plot, [string(r)], als_times * 1000, color=:cyan, outliers=false, label="CP-ALS")
            boxplot!(time_plot, [string(r)], gradient_times * 1000, color=:red, outliers=false, label="Gradient Descent")
            boxplot!(iteration_plot, [string(r)], als_iterations, color=:cyan, outliers=false, label="CP-ALS")
            boxplot!(iteration_plot, [string(r)], gradient_iterations, color=:red, outliers=false, label="Gradient Descent")
            boxplot!(allocation_per_iteration_plot, [string(r)], als_allocation_per_iteration, color=:cyan, outliers=false, label="CP-ALS")
            boxplot!(allocation_per_iteration_plot, [string(r)], gradient_allocation_per_iteration, color=:red, outliers=false, label="Gradient Descent")
            boxplot!(time_per_iteration_plot, [string(r)], als_time_per_iteration * 1000, color=:cyan, outliers=false, label="CP-ALS")
            boxplot!(time_per_iteration_plot, [string(r)], gradient_time_per_iteration * 1000, color=:red, outliers=false, label="Gradient Descent")
            first = false
        else
            boxplot!(time_plot, [string(r)], als_times * 1000, color=:cyan, outliers=false, label="")
            boxplot!(time_plot, [string(r)], gradient_times * 1000, color=:red, outliers=false, label="")
            boxplot!(iteration_plot, [string(r)], als_iterations, color=:cyan, outliers=false, label="")
            boxplot!(iteration_plot, [string(r)], gradient_iterations, color=:red, outliers=false, label="")
            boxplot!(allocation_per_iteration_plot, [string(r)], als_allocation_per_iteration, color=:cyan, outliers=false, label="")
            boxplot!(allocation_per_iteration_plot, [string(r)], gradient_allocation_per_iteration, color=:red, outliers=false, label="")
            boxplot!(time_per_iteration_plot, [string(r)], als_time_per_iteration * 1000, color=:cyan, outliers=false, label="")
            boxplot!(time_per_iteration_plot, [string(r)], gradient_time_per_iteration * 1000, color=:red, outliers=false, label="")
        end
        als_success_rate = sum(als_convergences) / length(als_convergences)
        gradient_success_rate = sum(gradient_convergences) / length(gradient_convergences)
        push!(als_success_rates, als_success_rate)
        push!(gradient_success_rates, gradient_success_rate)
        println("ALS Convergence for rank $r: $(sum(als_convergences)) out of $samples = $(sum(als_convergences)/samples * 100)%")
        println("Gradient Descent Convergence for rank $r: $(sum(gradient_convergences)) out of $samples = $(sum(gradient_convergences)/samples * 100)%")
    end
    #= plot!(allocation_per_iteration_plot, ranks, als_allocation_per_iteration_medians, label="CP-ALS", color=:blue)
    plot!(allocation_per_iteration_plot, ranks, gradient_allocation_per_iteration_medians, label="Gradient Descent", color=:red)
    savefig(allocation_per_iteration_plot, "allocation_per_iteration_comparison1-100.png") =#
    plot!(success_rate_plot, ranks, als_success_rates, label="CP-ALS", color=:cyan)
    plot!(success_rate_plot, ranks, gradient_success_rates, label="Gradient Descent", color=:red)
    savefig(time_per_iteration_plot, "time_per_iteration_to_rank_comparison1-10 dim=$m x $n x $p seed=$seed maxiters=$maxiters samples=$samples.png")
    savefig(time_plot, "time_to_rank_comparison1-10 dim=$m x $n x $p seed=$seed maxiters=$maxiters samples=$samples.png")
    savefig(iteration_plot, "iteration_to_rank_comparison1-10 dim=$m x $n x $p seed=$seed maxiters=$maxiters samples=$samples.png")
    savefig(allocation_per_iteration_plot, "allocation_per_iteration_to_rank_comparison1-10 dim=$m x $n x $p seed=$seed maxiters=$maxiters samples=$samples.png")
    savefig(success_rate_plot, "success_rate_to_rank_comparison1-10 dim=$m x $n x $p seed=$seed maxiters=$maxiters samples=$samples.png")
end

function time_to_size_comparison(;seed=1, rank=1, maxiters=5000, samples=50)
    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    time_plot = boxplot(xlabel="Tensor Dimension: RxRxR", ylabel="Execution Time(ms)", title="Execution Time over Dimension Comparison", legend=:topleft, yscale=:log10, outliers=false)
    iteration_plot = boxplot(xlabel="Tensor Dimension: RxRxR", ylabel="Iterations to Convergence", title="Iterations to Convergence over Dimension Comparison", legend=:topleft, yscale=:log10, outliers=false)
    success_rate_plot = plot(xlabel="Tensor Dimension: RxRxR", ylabel="Success Rate", title="Success Rate over Dimension Comparison", legend=:topleft, yscale=:linear)
    time_per_iteration_plot = boxplot(xlabel="Tensor Dimension: RxRxR", ylabel="Time per Iteration (ms)", title="Time per Iteration over Dimension Comparison", legend=:topleft, yscale=:log10, outliers=false)
    allocation_per_iteration_plot = boxplot(xlabel="Tensor Dimension: RxRxR", ylabel="Memory Allocated per Iteration (bytes)", title="Memory Allocated per Iteration over Dimension Comparison", legend=:topleft, yscale=:log10, outliers=false)
    
    first = true
    Random.seed!(seed)
    als_success_rates = Float64[]
    gradient_success_rates = Float64[]
    for dim in dimensions
        als_convergences = Bool[]
        gradient_convergences = Bool[]
        als_times = Float64[]
        gradient_times = Float64[]
        als_iterations = Int[]
        gradient_iterations = Int[]
        als_allocations = Int[]
        gradient_allocations = Int[]
        als_time_per_iteration = Float64[]
        gradient_time_per_iteration = Float64[]
        als_allocation_per_iteration = Float64[]
        gradient_allocation_per_iteration = Float64[]
        for i in 1:samples
            A0, B0, C0 = randn(dim, 1), randn(dim, 1), randn(dim, 1)
            test = generate_tensor(3, [dim, dim, dim], rank, reseed=false)
            initv = mats2vec(A0, B0, C0)
            test_full = construct_kruskal(test)
            als_result, als_time, als_allocated, _, _ = @timed cp_als_dway(test_full, 1, init=[A0, B0, C0], maxiters=maxiters)
            gradient_result, gradient_time, gradient_allocated, _, _ = @timed gradient_descent(test_full, 1, init=initv, maxiters=maxiters) 
            _, _, als_convergence, als_iteration, _ = als_result
            _, _, gradient_convergence, gradient_iteration, _ = gradient_result
            push!(als_convergences, als_convergence)
            push!(gradient_convergences, gradient_convergence)
            push!(als_times, als_time)
            push!(als_iterations, als_iteration)
            push!(als_allocations, als_allocated)
            if als_iteration > 0
                push!(als_time_per_iteration, als_time / als_iteration)
                push!(als_allocation_per_iteration, als_allocated / als_iteration)
            end
            
            push!(gradient_times, gradient_time)
            push!(gradient_iterations, gradient_iteration)
            push!(gradient_allocations, gradient_allocated)
                
            
            if gradient_iteration > 0
                push!(gradient_time_per_iteration, gradient_time / gradient_iteration)
                push!(gradient_allocation_per_iteration, gradient_allocated / gradient_iteration)
            end
        end
        als_success_rate = sum(als_convergences) / length(als_convergences)
        gradient_success_rate = sum(gradient_convergences) / length(gradient_convergences)
        push!(als_success_rates, als_success_rate)
        push!(gradient_success_rates, gradient_success_rate)
        if first
            boxplot!(time_plot, [string(dim)], als_times * 1000, color=:cyan, outliers=false, label="CP-ALS")
            boxplot!(time_plot, [string(dim)], gradient_times * 1000, color=:red, outliers=false, label="Gradient Descent")
            boxplot!(iteration_plot, [string(dim)], als_iterations, color=:cyan, outliers=false, label="CP-ALS")
            boxplot!(iteration_plot, [string(dim)], gradient_iterations, color=:red, outliers=false, label="Gradient Descent")
            boxplot!(time_per_iteration_plot, [string(dim)], als_time_per_iteration * 1000, color=:cyan, outliers=false, label="CP-ALS")
            boxplot!(time_per_iteration_plot, [string(dim)], gradient_time_per_iteration * 1000, color=:red, outliers=false, label="Gradient Descent")
            boxplot!(allocation_per_iteration_plot, [string(dim)], als_allocation_per_iteration, color=:cyan, outliers=false, label="CP-ALS")
            boxplot!(allocation_per_iteration_plot, [string(dim)], gradient_allocation_per_iteration, color=:red, outliers=false, label="Gradient Descent")
            first = false
        else
            boxplot!(time_plot, [string(dim)], als_times * 1000, color=:cyan, outliers=false, label="")
            boxplot!(time_plot, [string(dim)], gradient_times * 1000, color=:red, outliers=false, label="") 
            boxplot!(iteration_plot, [string(dim)], als_iterations, color=:cyan, outliers=false, label="")
            boxplot!(iteration_plot, [string(dim)], gradient_iterations, color=:red, outliers=false, label="")
            boxplot!(time_per_iteration_plot, [string(dim)], als_time_per_iteration * 1000, color=:cyan, outliers=false, label="")
            boxplot!(time_per_iteration_plot, [string(dim)], gradient_time_per_iteration * 1000, color=:red, outliers=false, label="")
            boxplot!(allocation_per_iteration_plot, [string(dim)], als_allocation_per_iteration, color=:cyan, outliers=false, label="")
            boxplot!(allocation_per_iteration_plot, [string(dim)], gradient_allocation_per_iteration, color=:red, outliers=false, label="")
            savefig(success_rate_plot, "success_rate_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")
            savefig(time_plot, "time_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")
            savefig(iteration_plot, "iteration_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")
            savefig(allocation_per_iteration_plot, "allocation_per_iteration_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")
            savefig(time_per_iteration_plot, "time_per_iteration_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")
        end
        println("Completed plotting for dimension: $dim x $dim x $dim, Rank: $rank")
    end

    plot!(success_rate_plot, dimensions, als_success_rates, label="CP-ALS", color=:cyan)
    plot!(success_rate_plot, dimensions, gradient_success_rates, label="Gradient Descent", color=:red)
    savefig(success_rate_plot, "success_rate_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")
    savefig(time_plot, "time_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")
    savefig(iteration_plot, "iteration_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")
    savefig(allocation_per_iteration_plot, "allocation_per_iteration_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")
    savefig(time_per_iteration_plot, "time_per_iteration_to_dimension_comparison1-150 rank=$rank seed=$seed samples=$samples maxiters=$maxiters.png")

end


time_to_size_comparison(seed=7408, rank=4, samples=20, maxiters=1000);






