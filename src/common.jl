abstract type LLM end

@with_kw struct BlackBoxState
    prompt::String
    suffix::String = ""
    response::String = ""
    score::Float64 = 0
end

BlackBoxState(prompt::String, suffix::String, response::String) = BlackBoxState(; prompt, suffix, response)

@with_kw mutable struct BlackBoxParams
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
