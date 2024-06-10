using Kov
using BSON

targets = [
    # (name="gpt3", model=gpt_model("gpt-3.5-turbo")),
    # (name="gpt4", model=gpt_model("gpt-4-1106-preview")),
    # (name="vicuna-7b", model=vicuna_model("vicuna-7b-v1.5")), # ! NOTE: Set up the server.
    # (name="vicuna-13b", model=vicuna_model("vicuna-13b-v1.5")), # ! NOTE: Set up the server.
    # (name="zephyr-7b", model=vicuna_model("zephyr-7b-alpha")), # ! NOTE: Set up the server.
    # (name="pythia-6.9b", model=vicuna_model("pythia-6.9b")), # ! NOTE: Set up the server.
    # (name="chatglm-6b", model=vicuna_model("chatglm-6b")), # ! NOTE: Set up the server.
    # (name="openchat_3.5", model=vicuna_model("openchat_3.5")), # ! NOTE: Set up the server.
    (name="fastchat-3b", model=vicuna_model("fastchat-t5-3b-v1.0")), # ! NOTE: Set up the server.
    # (name="llama-2", model=llama_model("llama-2-7b-chat-hf"; is_local=true)), # ! NOTE: Set up the server.
]

function show_results(results::Dict)
    for target in targets
        name = target.name
        @info """
        $name
        Average score: $(mean(idx->mean(results[name][idx][:scores]), 1:5))
        Maximum score: $(maximum(idx->maximum(results[name][idx][:scores]), 1:5))
        """
    end
end

function run_transfer(fn_training_data::Function, targets; filename="results-transfer.bson", wait_on_localhost=true, is_baseline=false, n=10, temperature=0)
    results = Dict()
    for target in targets
        @info target.name
        if wait_on_localhost && target.name in ["vicuna-7b", "vicuna-13b", "llama-2", "zephyr-7b", "pythia-6.9b", "openchat_3.5", "fastchat-3b"]
            print("Please press [ENTER] when the $(target.name) model is loaded to localhost...")
            readline()
        end
        results[target.name] = test_transfer(fn_training_data, target.model; use_mean=false, is_baseline, n, temperature)
    end
    BSON.@save filename results
    show_results(results)
    return results
end

temperature = 0.1
name = targets[1].name

## Kov-Adversarial.
fn_training_data = i->"gpt3-advbench$i-adv-mdp-data.bson"
kov_adv_results = run_transfer(fn_training_data, targets; filename="kov-adv-results-transfer-$name.bson", temperature)

## Kov-Aligned.
fn_training_data = i->"gpt3-advbench$i-aligned-mdp-data.bson"
kov_ali_results = run_transfer(fn_training_data, targets; filename="kov-ali-results-transfer-$name.bson", temperature)

## Baseline.
fn_training_data = i->missing # "gpt3-advbench$i-baseline-mdp-data.bson"
kov_ali_results = run_transfer(fn_training_data, targets; is_baseline=true, filename="baseline-results-transfer-$name.bson", temperature)

## GCG:
fn_training_data = i->"gpt3-advbench$i-gcg-data.bson"
gcg_results = run_transfer(fn_training_data, targets; filename="gcg-results-transfer-$name.bson", temperature)
