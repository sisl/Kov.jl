using Kov
using StatsBase
using BSON

names = [
    "vicuna-7b",
    "vicuna-13b",
    "llama-2",
    "gpt3",
    "gpt4"
]

function evaluate_suffixes(load_data::Function)
    baseline_results = Dict()
    for name in names
        @info name
        avg_scores = []
        for benchmark_idx in 1:5
            @info benchmark_idx
            scores = load_data(name, benchmark_idx)
            push!(avg_scores, mean(scores))
        end
        baseline_results[name] = mean_and_std(avg_scores)
    end
    return baseline_results
end


## Kov-Adv.
load_data = (name, benchmark_idx)->begin
    data = BSON.load("$name-vicuna7b-kov-value-gcg-advbench$benchmark_idx-adv-mdp-data.bson")[:data]
    responses = map(d->d.response, data)
    scores = map(Kov.score, responses)
    return scores
end
kov_adv_eval = evaluate_suffixes(load_data)


## Kov-Aligned.
load_data = (name, benchmark_idx)->begin
    data = BSON.load("$name-vicuna7b-kov-value-gcg-aligned-advbench$benchmark_idx-aligned-mdp-data.bson")[:data]
    responses = map(d->d.response, data)
    scores = map(Kov.score, responses)
    return scores
end
kov_ali_eval = evaluate_suffixes(load_data)


## GCG.
# Reload and re-save data into proper transfer format
load_data = (benchmark_idx)->begin
    suffixes = BSON.load("vicuna-7b-gcg-results-advbench$benchmark_idx")[:results][:suffixes]
    prompt = load_prompts(benchmark_idx)
    data = [(prompt=prompt, suffix=suffix) for suffix in suffixes]
    BSON.@save "vicuna-7b-gcg-results-advbench$benchmark_idx-data.bson" data
    data
end
gcg_eval = evaluate_suffixes(load_data)

