using Revise
using Kov
using BSON
using Random
using Plots; default(fontfamily="Computer Modern", framestyle=:box)

if @isdefined(surrogate)
    surrogate.params.seed = 0
else
    model_path = expanduser("~/vicuna/vicuna-7b-v1.5")

    whitebox_params = WhiteBoxParams(;
        model_path,
        num_steps=100,
        n_tokens=8,
        break_on_success=false,
        include_perp=true,
        device="cuda:0",
    )
    surrogate = WhiteBoxMDP(whitebox_params; device_map=nothing)
end

for benchmark_idx in 1:5
    global surrogate, whitebox_params, final_state, final_suffix
    @info "Benchmark index $benchmark_idx (GCG)"

    Kov.change_benchmark!(surrogate, benchmark_idx)
    Random.seed!(surrogate.params.seed)
    name = "gpt3-advbench$benchmark_idx-gcg-data.bson"

    Kov.WhiteBox.clear!()
    empty!(surrogate.data)
    if surrogate.params.include_perp
        setgcg!(surrogate) # GCG
    end

    ## Run NGCG
    final_state, losses, successes = optimize(surrogate; return_logs=true)
    final_suffix = final_state.adv_suffix

    suffixes = map(d->d.suffix, surrogate.data)

    data = [(; prompt=surrogate.params.prompt, suffix, loss, score=-loss, success) for (suffix, loss, success) in zip(suffixes, losses, successes)]

    BSON.@save name data
end
