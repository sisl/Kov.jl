abstract type LLM end

@with_kw struct BlackBoxState
    prompt::String
    suffix::String = ""
    response::String = ""
    score::Float64 = 0
end

BlackBoxState(prompt::String, suffix::String, response::String) = BlackBoxState(; prompt, suffix, response)

@with_kw mutable struct BlackBoxParams
    solver = DPWSolver(n_iterations=10,
                       depth=2,
                       check_repeat_action=true,
                       exploration_constant=1.0,
                       k_action=2.0,
                       alpha_action=0.0,
                       enable_state_pw=false,
                       tree_in_info=true,
                       show_progress=true,
                       estimate_value=0)
    max_steps = 10 # maximum white-box MDP environment steps when calling `POMDPTools.stepthrough`
    temperature = 0
    verbose = true
    verbose_critical = true
    critical_threshold = 0.05
end

@with_kw mutable struct BlackBoxMDP <: MDP{BlackBoxState,String}
    params::BlackBoxParams = BlackBoxParams()
    target_model::LLM
    surrogate::WhiteBoxMDP
    prompt::String
    flipped::Bool = false # move to params.
    data::Vector = []
end

BlackBoxMDP(target_model::LLM, surrogate::WhiteBoxMDP, prompt::String) = BlackBoxMDP(; target_model, surrogate, prompt)
BlackBoxMDP(params::BlackBoxParams, target_model::LLM, surrogate::WhiteBoxMDP, prompt::String) = BlackBoxMDP(; params, target_model, surrogate, prompt)
