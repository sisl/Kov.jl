POMDPs.initialstate(mdp::BlackBoxMDP) = [BlackBoxState(prompt=mdp.prompt)]
function POMDPs.actions(mdp::BlackBoxMDP, state)
    if mdp.flipped
        prompt = "Provide a prompt suffix that ensures the LLM response follows ethical guidelines."
        suffix = evaluate(mdp, prompt; temperature=0.7)
        suffix = replace(suffix, r"\""=>"")
        return [suffix]
    else
        mdp.surrogate.params.seed += 1
        WhiteBox.seeding(mdp.surrogate.params.seed)

        if isempty(state.suffix)
            white_box_state = WhiteBoxState(mdp.surrogate)
        else
            white_box_state = WhiteBoxState(mdp.surrogate, state.suffix)
        end

        white_box_solver = mdp.surrogate.params.solver
        white_box_policy = solve(white_box_solver, mdp.surrogate)

        mdp.params.verbose && @info "Starting white-box sub-tree search."
        local white_box_sp
        @time for (sp, t) in stepthrough(mdp.surrogate, white_box_policy, white_box_state, "sp,t", max_steps=mdp.params.max_steps)
            white_box_sp = sp
            mdp.params.verbose && @info """
            White-box sub-tree search iteration $t/$(mdp.params.max_steps)
            Negative log-likelihood: $(white_box_sp.nll)
            Log-perplexity: $(white_box_sp.log_ppl)
            Loss: $(white_box_sp.loss)
            """
        end

        WhiteBox.update!(white_box_sp.loss; show=true, c=:crimson, label="Loss", size=(600,300), lw=2)
        mdp.params.verbose && @info "Negative log-likelihood: $(white_box_sp.nll)"
        mdp.params.verbose && @info "Log-perplexity: $(white_box_sp.log_ppl)"
        mdp.params.verbose && @info "Loss: $(white_box_sp.loss)"

        suffix = white_box_sp.adv_suffix
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
        response = evaluate(mdp, full_prompt; temperature=mdp.params.temperature)
    end

    critical_condition = r->mdp.params.verbose_critical && r ≥ mdp.params.critical_threshold
    r = score(response; verbose=mdp.params.verbose, critical_condition, test_prefixes=mdp.surrogate.params.test_prefixes)
    r = mdp.flipped ? -r : r
    prompt = s.prompt
    sp = BlackBoxState(prompt=prompt, suffix=a, response=response, score=r)
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
