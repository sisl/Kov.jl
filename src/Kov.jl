module Kov

include(joinpath("white_box", "WhiteBox.jl"))

using Reexport
using Random
using Statistics
using Parameters
using Pkg
using CSV
using BSON
using PrettyTables
using DataFrames
@reexport using POMDPs
@reexport using PyCall
@reexport using .WhiteBox

export
    BlackBoxParams,
    BlackBoxMDP,
    gpt_model,
    llama_model,
    guanaco_model,
    vicuna_model,
    evaluate,
    completion,
    moderation,
    avg_moderated,
    avg_moderated_score,
    moderated_scores,
    moderated_flags,
    score,
    create_suffix,
    compute_moderation,
    get_scores,
    run_baseline,
    select_action,
    value_estimate,
    transfer,
    test_transfer,
    print_box

global Perspective, OpenAI, PeftModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, torch

# https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages?language=en_US
global PERSPECTIVE_OUTPUTS = [
    "TOXICITY",
    "INFLAMMATORY",
    "OBSCENE",
    "SEVERE_TOXICITY",
    "INSULT",
    "SEXUALLY_EXPLICIT",
    "PROFANITY",
    "THREAT",
    "IDENTITY_ATTACK"
]

include("common.jl")
include("llm.jl")
include("mdp.jl")
include("transfer.jl")
include("baseline.jl")

function __init__()
    if Sys.iswindows()
        ENV["PYTHON"] = "C:/Python310/python.exe"
        Pkg.build("PyCall")
    end

    perspective = pyimport("perspective")
    perspective_api_key = ENV["PERSPECTIVE_API_KEY"]
    global Perspective = perspective.PerspectiveAPI(perspective_api_key)

    openai = pyimport("openai")
    global OpenAI = openai.OpenAI

    transformers = pyimport("transformers")
    global AutoModelForCausalLM = transformers.AutoModelForCausalLM
    global AutoTokenizer = transformers.AutoTokenizer
    global BitsAndBytesConfig = transformers.BitsAndBytesConfig

    global torch = pyimport("torch")

    return nothing
end

end # module Kov
