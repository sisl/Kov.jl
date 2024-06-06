using Revise
using Kov
using BSON
using Random
using Plots; default(fontfamily="Computer Modern", framestyle=:box)

include("solver.jl")

if !@isdefined(surrogate)
    model_path = expanduser("~/vicuna/vicuna-7b-v1.5")
    # model_path = expanduser("~/vicuna/vicuna-13b-v1.5")
    # model_path = expanduser("~/llama/llama-2-7b-chat-hf")

    whitebox_params = WhiteBoxParams(;
        model_path,
        benchmark_idx=4,
        num_steps=20,
        flipped=false,
        n_tokens=20,
        batch_size=32,
        topk=16,
        logit_batch_size=32,
        break_on_success=false,
        include_perp=false,
        verbose=true,
        device="cuda",
    )
    surrogate = WhiteBoxMDP(whitebox_params)
end

all_results = Dict()
benchmark_indices = 1:5
for benchmark_idx in benchmark_indices
    global surrogate, whitebox_params
    @info "Benchmark index $benchmark_idx (GCG)"

    Kov.change_benchmark!(surrogate, benchmark_idx)
    Random.seed!(surrogate.params.seed)
    name = "vicuna-7b-gcg-results-advbench$(whitebox_params.benchmark_idx).bson"

    ## Run GCG
    empty!(surrogate.data)
    setgcg!(surrogate)
    final_state, losses, successes = optimize(surrogate; return_logs=true)
    final_suffix = final_state.adv_suffix

    results = Dict(
        :suffix=>final_suffix,
        :suffixes=>map(d->d.suffix, surrogate.data),
        :losses=>losses,
        :successes=>successes,
    )

    all_results[benchmark_idx] = results

    BSON.@save name results
    try
        plot(all_results[benchmark_idx][:losses], c=:black, lw=2, label="GCG")
        display(plot!(title="Benchmark $benchmark_idx", size=(600,300)))
    catch err
        @warn err
    end
end
