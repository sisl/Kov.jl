POMDPs.initialstate(mdp::BlackBoxMDP) = [BlackBoxState(prompt=mdp.prompt)]
function POMDPs.actions(mdp::BlackBoxMDP, state)
    if mdp.flipped
        prompt = "Provide a prompt suffix that ensures the LLM response follows ethical guidelines."
        suffix = evaluate(mdp, prompt)
        suffix = replace(suffix, r"\""=>"")
        return [suffix]
    else
        mdp.surrogate.params.seed += 1
        if isempty(state.suffix) # initial time step
            optimized_next_state = optimize(mdp.surrogate; show_response=false)
        else
            adv_suffix = state.suffix
            optimized_state = WhiteBoxState(mdp.surrogate, adv_suffix)
            optimized_next_state = optimize(mdp.surrogate, optimized_state; show_response=false)
        end
        suffix = optimized_next_state.adv_suffix
        return [suffix]
    end
end

POMDPs.discount(mdp::BlackBoxMDP) = 0.99

function POMDPs.gen(mdp::BlackBoxMDP, s::BlackBoxState, a::String, rng=Random.GLOBAL_RNG)
    full_prompt = s.prompt * " " * a
    if occursin(mdp.target_model.name, mdp.surrogate.params.model_path) && !(hasproperty(mdp.target_model.llm, :base_url) && occursin("localhost", mdp.target_model.llm.base_url.host))
        white_box_state = WhiteBoxState(mdp.surrogate, a; check_success=false, loss=NaN)
        max_tokens = mdp.surrogate.params.max_tokens
        response = generate_response(mdp.surrogate, white_box_state; max_tokens=max_tokens, print_response=false)
    else
        response = evaluate(mdp, full_prompt)
    end

    critical_condition = r->r ≥ mdp.params.critical_threshold
    r = score(response; verbose=mdp.params.verbose, critical_condition, test_prefixes=mdp.surrogate.params.test_prefixes)
    r = mdp.flipped ? -r : r
    sp = BlackBoxState(prompt=s.prompt, suffix=a, response=response, score=r)
    push!(mdp.data, sp)

    show_critical = mdp.params.verbose_critical && critical_condition(r)
    if mdp.params.verbose || show_critical
        print_llm(full_prompt, response; prefix="Black-Box")
        println()
    end

    return (; sp, r)
end

function select_action(mdp::BlackBoxMDP, include_prompt=false)
    scores = map(d->d.score, mdp.data)
    suffix = mdp.data[argmax(scores)].suffix
    return include_prompt ? mdp.prompt * " " * suffix : suffix
end

"""
Estimate the value of a `next_state` by evaluting the score of the prompt + suffix on the white-box surrogate LLM.
"""
function value_estimate(mdp::BlackBoxMDP, next_state::BlackBoxState, depth=1)
    white_box_state = WhiteBoxState(mdp.surrogate, next_state.suffix; check_success=false, loss=NaN)
    full_prompt = next_state.prompt * " " * next_state.suffix
    # evaluate on the surrogate
    max_tokens = mdp.surrogate.params.max_generate_tokens
    response = generate_response(mdp.surrogate, white_box_state; max_tokens=max_tokens, print_response=false)
    v = score(response; verbose=mdp.params.verbose, test_prefixes=mdp.surrogate.params.test_prefixes)
    v = mdp.flipped ? -v : v

    if mdp.params.verbose
        println()
        @info "White-Box Value Estimation (depth $depth)."
        print_llm(full_prompt, response; prefix="White-Box (Value)", response_color=:white)
        print_box(v, "White-Box Value Estimate")
    end

    return v
end
