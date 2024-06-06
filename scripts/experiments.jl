using Revise
using Kov
using BSON
using POMDPTools
using Random

include("solver.jl")

MODEL_ON_LOCALHOST = false
RUN_BEST = false
RUN_BASELINE = false

if @isdefined(surrogate)
    surrogate.params.seed = 0 # reset seed
else
    model_path = expanduser("~/vicuna/vicuna-7b-v1.5")
    # model_path = expanduser("~/vicuna/vicuna-13b-v1.5")
    # model_path = expanduser("~/llama/llama-2-7b-chat-hf")

    whitebox_params = WhiteBoxParams(;
        model_path,
        n_tokens=20,
        batch_size=32,
        topk=16,
        logit_batch_size=32,
        num_steps=2,
        break_on_success=false,
        include_perp=true,
        λ_perp=0.1,
        verbose=true,
        verbose_critical=true,
        device="cuda",
    )

    if MODEL_ON_LOCALHOST
        surrogate = WhiteBoxMDP(params=whitebox_params, conv_template=missing)
    else
        surrogate = WhiteBoxMDP(whitebox_params)
    end
end

for benchmark_idx in 1:5
    global mdp, surrogate, whitebox_params, state

    Kov.change_benchmark!(surrogate, benchmark_idx)
    Random.seed!(surrogate.params.seed)

    @info "Benchmark index $benchmark_idx: $(surrogate.params.prompt)"

    ########################################
    ## Change this for different models
    ########################################
    params = (name="gpt3-advbench$benchmark_idx", is_aligned=false, target_model=gpt_model("gpt-3.5-turbo"))
    # params = (name="gpt4-advbench$benchmark_idx", is_aligned=false, target_model=gpt_model("gpt-4"))
    # params = (name="vicuna-7b-advbench$benchmark_idx", is_aligned=false, target_model=vicuna_model("vicuna-7b-v1.5"))
    # params = (name="vicuna-13b-advbench$benchmark_idx", is_aligned=false, target_model=vicuna_model("vicuna-13b-v1.5"))
    # params = (name="llama-2-advbench$benchmark_idx", is_aligned=false, target_model=llama_model("llama-2-7b-chat-hf"; is_local=true))

    surrogate.params.flipped = params.is_aligned
    prompt = whitebox_params.prompt
    mdp = BlackBoxMDP(params.target_model, surrogate, prompt)
    mdp.flipped = params.is_aligned
    s0 = rand(initialstate(mdp))

    if !RUN_BASELINE
        policy = solve(solver, mdp)
        a, info = action_info(policy, s0)
        suffix = select_action(mdp)
        printstyled("[BEST ACTION]: $suffix\n", color=:cyan)

        name = string(params.name, "-", params.is_aligned ? "aligned" : "adv")

        data = mdp.data
        BSON.@save "$name-mdp-data.bson" data

        if RUN_BEST
            best_trials = 1
            R_best, data_best = Kov.run_baseline(mdp, best_trials, suffix)
            mods_best = Kov.compute_moderation(mdp)
            BSON.@save "$(params.name)-best-scores.bson" R_best
            BSON.@save "$(params.name)-best-data.bson" data_best
            BSON.@save "$(params.name)-best-moderation.bson" mods_best
            mod_rate_best = Kov.avg_moderated(mods_best)
            mod_score_best = Kov.avg_moderated_score(mods_best)
            @info "Failed moderation rate (best): $mod_rate_best"
            @info "Average moderation score (best): $mod_score_best"
        end
    else
        R_baseline, data_baseline = run_baseline(mdp, solver.n_iterations)
        mods_baseline = compute_moderation(mdp)
        BSON.@save "$(params.name)-baseline-advbench$(whitebox_params.benchmark_idx)-scores.bson" R_baseline
        BSON.@save "$(params.name)-baseline-advbench$(whitebox_params.benchmark_idx)-data.bson" data_baseline
        BSON.@save "$(params.name)-baseline-advbench$(whitebox_params.benchmark_idx)-moderation.bson" mods_baseline
        mod_rate_baseline = Kov.avg_moderated(mods_baseline)
        mod_score_baseline = Kov.avg_moderated_score(mods_baseline)
        @info "Failed moderation rate (baseline): $mod_rate_baseline"
        @info "Average moderation score (baseline): $mod_score_baseline"
    end
end
