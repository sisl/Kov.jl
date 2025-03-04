module WhiteBox

include(joinpath(@__DIR__, "..", "LivePlots.jl"))

using .LivePlots
using Reexport
using Random
using Statistics
using Parameters
using Pkg
using Plots
using CSV
using DataFrames
using MCTS
@reexport using PrettyTables
@reexport using POMDPs
@reexport using PyCall

export
    WhiteBoxParams,
    WhiteBoxMDP,
    WhiteBoxState,
    optimize,
    generate_response,
    seeding,
    change_benchmark!,
    load_prompts,
    load_targets,
    setgcg!,
    setarca!,
    change_placeholder!,
    print_box,
    print_llm,
    sample_tokens

global torch, np

include("data.jl")
include("common.jl")
include("mdp.jl")
include("plotting.jl")

function __init__()
    if Sys.iswindows()
        ENV["PYTHON"] = "C:/Python310/python.exe"
        Pkg.build("PyCall")
        py"""
        import sys
        sys.path.insert(0, $(joinpath(@__DIR__, "..")))
        """
    else
        pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "..", ".."))
    end

    py"""
    import gc
    from python.gcg.utils import *
    from python.gcg.string_utils import *
    """

    global load_model_and_tokenizer = py"load_model_and_tokenizer"
    global load_conversation_template = py"load_conversation_template"
    global SuffixManager = py"SuffixManager"
    global get_nonascii_toks = py"get_nonascii_toks"
    global token_gradients = py"token_gradients"
    global sample_control = py"sample_control"
    global get_filtered_cands = py"get_filtered_cands"
    global get_logits = py"get_logits"
    global target_loss = py"target_loss"
    global check_for_attack_success = py"check_for_attack_success"
    global generate = py"generate"
    global arca_suffix = py"arca_suffix"
    global filter_forbidden_toks = py"filter_forbidden_toks"

    global torch = pyimport("torch")
    global np = pyimport("numpy")

    return nothing
end


"""
Set RNG seeds for numpy, torch, and CUDA (if using GPU).
"""
function seeding(seed=20)
    Random.seed!(seed) # Julia
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed_all(seed) # CUDA (GPU)
end


function optimize(mdp::WhiteBoxMDP, state=WhiteBoxState(mdp); show_response=true, return_logs=false)
    params = mdp.params
    seeding(params.seed)
    losses = []
    successes = []

    for i in 1:params.num_steps
        if mdp.params.verbose
            println()
            @info "Greedy Coordinate Gradient ($i/$(params.num_steps))"
        end
        act = rand(actions(mdp, state))
        (state, r) = @gen(:sp, :r)(mdp, state, act)
        loss = -r
        is_success = state.is_success
        push!(losses, loss)
        push!(successes, is_success)

        # Create a dynamic plot for the loss.
        plot_loss(losses; flipped=mdp.params.flipped)

        # Notice that for the purpose of demo we stop immediately if we pass the checker but you are free to
        # comment this to keep the optimization running for longer (to get a lower loss).
        if is_success && params.break_on_success
            break
        end
    end

    # generate jailbroken response
    show_response && generate_response(mdp, state)

    if return_logs
        return state, losses, successes
    else
        return state
    end
end


"""
Set parameters to match the GCG algorithm.
(i.e., do not select the top-t tokens, and use a uniform distribution when selecting new controls)
"""
function setgcg!(mdp::WhiteBoxMDP)
    mdp.params.topt = 1
    mdp.params.m_tokens = 1
    mdp.params.use_uniform = true
    mdp.params.use_arca = false
    mdp.params.include_perp = false
end


"""
Set parameters to use the ARCA algorithm.
(i.e., do not select the top-t tokens, and use the ARCA method to select the candidate action suffix)
"""
function setarca!(mdp::WhiteBoxMDP)
    mdp.params.topt = 1
    if mdp.params.topk > 64
        @warn "ARCA: top-k > 64 may cause GPU out-of-memory errors (top-k = $(mdp.params.topk))."
    end
    mdp.params.use_arca = true
end


function generate_response(mdp::WhiteBoxMDP, state; print_response=true, max_tokens=mdp.params.max_generate_tokens, temperature=nothing, top_p=nothing, do_sample=false)
    params = mdp.params
    model, tokenizer, suffix_manager = mdp.model, mdp.tokenizer, state.suffix_manager
    adv_suffix = state.adv_suffix
    input_ids = suffix_manager.get_input_ids(adv_suffix).to(params.device)

    gen_config = model.generation_config
    gen_config.max_new_tokens = max_tokens
    gen_config.do_sample = do_sample
    gen_config.temperature = temperature
    gen_config.top_p = top_p

    completion = strip(tokenizer.decode((generate(model, tokenizer, input_ids, suffix_manager._assistant_role_slice, gen_config=gen_config)), skip_special_tokens=true))
    print_response && printstyled("\n[Completion]: $completion\n", color=:blue)
    return completion
end


end # module WhiteBox
