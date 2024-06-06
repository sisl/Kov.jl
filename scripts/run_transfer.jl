using Kov

targets = [
    (name="gpt3", model=gpt_model("gpt-3.5-turbo")),
    (name="gpt4", model=gpt_model("gpt-4")),
    (name="vicuna-7b", model=vicuna_model("vicuna-7b-v1.5")), # ! NOTE: Set up the server.
    (name="vicuna-13b", model=vicuna_model("vicuna-13b-v1.5")), # ! NOTE: Set up the server.
    (name="llama-2", model=llama_model("llama-2-7b-chat-hf"; is_local=true)), # ! NOTE: Set up the server.
]

function show_results(results::Dict)
    for target in targets
        name = target.name
        @show mean(idx->mean(results[name][idx][:scores]), 1:5)
    end
end

function run_transfer(fn_training_data::Function, targets; filename="results-transfer.bson", wait_on_localhost=true)
    results = Dict()
    for target in targets
        @info target.name
        if wait_on_localhost && target.name in ["vicuna-7b", "vicuna-13b", "llama-2"]
            print("Please press [ENTER] when the $(target.name) model is loaded to localhost...")
            readline()
        end
        results[target.name] = test_transfer(fn_training_data, target.model)
    end
    BSON.@save filename results
    show_results(results)
    return results
end

## Kov-Adversarial.
fn_training_data = i->"gpt3-vicuna7b-kov-value-gcg-advbench$i-adv-mdp-data.bson"
kov_adv_results = run_transfer(fn_training_data, targets; filename="kov-adv-results-transfer.bson")

## Kov-Aligned.
fn_training_data = i->"gpt3-vicuna7b-kov-value-gcg-aligned-advbench$i-aligned-mdp-data.bson"
kov_ali_results = run_transfer(fn_training_data, targets; filename="kov-ali-results-transfer.bson")

## GCG:
fn_training_data = i->"vicuna-7b-gcg-results-advbench$i-data.bson"
gcg_results = run_transfer(fn_training_data, targets; filename="gcg-results-transfer.bson")
