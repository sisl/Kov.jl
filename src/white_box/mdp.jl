function initialize_suffix_manager(mdp::WhiteBoxMDP)
    return SuffixManager(tokenizer=mdp.tokenizer,
                        conv_template=mdp.conv_template,
                        instruction=mdp.params.prompt,
                        target=mdp.params.target,
                        adv_string=mdp.params.adv_string_init)
end


WhiteBoxState(mdp::WhiteBoxMDP) = WhiteBoxState(mdp, mdp.params.adv_string_init)
function WhiteBoxState(mdp::WhiteBoxMDP, adv_suffix; check_success=false, loss=NaN, nll=NaN, log_ppl=NaN)
    params, model, tokenizer = mdp.params, mdp.model, mdp.tokenizer

    suffix_manager = initialize_suffix_manager(mdp)

    # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
    input_ids = suffix_manager.get_input_ids(adv_suffix)
    input_ids = input_ids.to(mdp.params.device)

    if check_success
        torch.set_grad_enabled(false)
        is_success = check_for_attack_success(model,
                                            tokenizer,
                                            input_ids,
                                            suffix_manager._assistant_role_slice,
                                            params.test_prefixes)
        torch.set_grad_enabled(true)
    else
        is_success = false
    end

    return WhiteBoxState(adv_suffix, suffix_manager, input_ids, is_success, loss, nll, log_ppl)
end


function POMDPs.initialstate(mdp::WhiteBoxMDP)
    adv_suffix = mdp.params.adv_string_init
    state = WhiteBoxState(mdp, adv_suffix)
    return [state]
end


function POMDPs.actions(mdp::WhiteBoxMDP, state::WhiteBoxState)
    params, model, tokenizer = mdp.params, mdp.model, mdp.tokenizer

    local new_adv_suffix
    try
        if params.use_arca
            new_adv_suffix = arca_suffix(model,
                                        tokenizer,
                                        state.input_ids,
                                        state.suffix_manager._control_slice,
                                        state.suffix_manager._target_slice,
                                        state.suffix_manager._loss_slice,
                                        not_allowed_tokens=mdp.not_allowed_tokens,
                                        k=params.topk)
            # GCG will select from a vector of candidates, while ARCA finds the best single candidate during this function call.
            new_adv_suffix = [new_adv_suffix]
        else
            # Step 2. Compute Coordinate Gradient
            coordinate_grad = token_gradients(model,
                                            state.input_ids,
                                            state.suffix_manager._control_slice,
                                            state.suffix_manager._target_slice,
                                            state.suffix_manager._loss_slice)

            # Step 3. Sample a batch of new tokens based on the coordinate gradient.
            # Notice that we only need the one that minimizes the loss.
            torch.set_grad_enabled(false)

            # Step 3.1 Slice the input to locate the adversarial suffix.
            adv_suffix_tokens = state.input_ids.__getitem__(state.suffix_manager._control_slice).to(params.device)

            # Step 3.2 Randomly sample a batch of replacements.
            new_adv_suffix_toks = sample_control(adv_suffix_tokens,
                                                coordinate_grad,
                                                params.batch_size,
                                                topk=params.topk,
                                                temp=1,
                                                not_allowed_tokens=mdp.not_allowed_tokens,
                                                flipped=params.flipped,
                                                uniform=params.use_uniform,
                                                m_tokens=params.m_tokens)

            # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
            # This step is necessary because tokenizers are not invertible
            # so Encode(Decode(tokens)) may produce a different tokenization.
            # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
            new_adv_suffix = get_filtered_cands(tokenizer,
                                                new_adv_suffix_toks,
                                                filter_cand=true,
                                                curr_control=state.adv_suffix,
                                                clean_up_tokenization_spaces=occursin("stablelm", lowercase(params.template_name)) ? false : nothing)

            torch.set_grad_enabled(true)

            # (Optional) Clean up the cache.
            py"""
            del $coordinate_grad, $adv_suffix_tokens
            """
            cleanup()
        end
    catch err
        torch.set_grad_enabled(true)
        rethrow(err)
    end

    return [new_adv_suffix]
end


function POMDPs.gen(mdp::WhiteBoxMDP, state::WhiteBoxState, act, rng=Random.GLOBAL_RNG)
    params, model, tokenizer = mdp.params, mdp.model, mdp.tokenizer

    torch.set_grad_enabled(false)

    local loss, best_new_adv_suffix, nll, log_ppl
    try
        # Step 3.4 Compute loss on these candidates and take the argmin.
        logits, ids = get_logits(model=model,
                                tokenizer=tokenizer,
                                input_ids=state.input_ids,
                                control_slice=state.suffix_manager._control_slice,
                                test_controls=act,
                                return_ids=true,
                                batch_size=params.logit_batch_size) # decrease this number if you run into OOM.

        losses, log_ppls = target_loss(logits,
                                    ids,
                                    state.suffix_manager._target_slice,
                                    state.suffix_manager._control_slice,
                                    state.suffix_manager._goal_slice,
                                    include_perp=params.include_perp,
                                    lambda_perp=params.λ_perp,
                                    flipped_perp=params.flipped,
                                    return_separate=true)

        if params.include_perp
            combined_losses = losses + log_ppls
        else
            combined_losses = losses
        end

        if params.flipped
            best_new_adv_suffix_id = combined_losses.argmax() # maximize the negative log-likelihood (i.e., minimize the likelihood)
        else
            best_new_adv_suffix_id = combined_losses.argmin() # minimize the negative log-likelihood (i.e., maximize the likelihood)
        end

        if params.topt == 1
            # Single top-token
            best_new_adv_suffix = py"$act[$best_new_adv_suffix_id]"
            loss = combined_losses.__getitem__(best_new_adv_suffix_id).detach().cpu().numpy()[1]
            nll = losses.__getitem__(best_new_adv_suffix_id).detach().cpu().numpy()[1]
            log_ppl = log_ppls.__getitem__(best_new_adv_suffix_id).detach().cpu().numpy()[1] / params.λ_perp
        else
            # Multiple top-t tokens
            best_new_adv_suffix_ids = torch.topk(losses, params.topt, largest=params.flipped)[2]
            best_new_adv_suffix_ids = best_new_adv_suffix_ids.detach().cpu().numpy()
            placeholder = tokenizer.encode(params.placeholder_token)[end]
            top_tokens = map(id->id.__getitem__(state.suffix_manager._control_slice), ids.__getitem__(best_new_adv_suffix_ids))
            top_tokens = map(t->t.detach().cpu().numpy(), top_tokens)
            combined = copy(top_tokens[end])
            for i in reverse(1:length(top_tokens)-1)
                for j in eachindex(combined)
                    if combined[j] != top_tokens[i][j] && top_tokens[i][j] != placeholder
                        combined[j] = top_tokens[i][j]
                    end
                end
            end
            best_new_adv_suffix = tokenizer.decode(combined)
            loss = combined_losses.__getitem__(best_new_adv_suffix_ids).mean().detach().cpu().numpy()[1]
            nll = losses.__getitem__(best_new_adv_suffix_ids).mean().detach().cpu().numpy()[1]
            log_ppl = log_ppls.__getitem__(best_new_adv_suffix_ids).mean().detach().cpu().numpy()[1] / params.λ_perp
        end

        torch.set_grad_enabled(true)
    catch err
        torch.set_grad_enabled(true)
        rethrow(err)
    end

    sp = WhiteBoxState(mdp, best_new_adv_suffix; check_success=true, loss, nll, log_ppl)
    r = params.flipped ? loss : -loss # NOTE for MDPs: maximize reward == minimize loss (but we want to maximize the loss in the flipped case)

    show_critical = mdp.params.verbose_critical && sp.is_success
    if params.verbose || show_critical
        full_prompt = mdp.params.prompt * " " * sp.adv_suffix
        println()
        if params.show_response
            response = generate_response(mdp, sp; print_response=false)
            print_llm(full_prompt, response; prefix="White-Box", response_color=:white)
        else
            high_prompt = Highlighter(
                (data,i,j)->true,
                foreground = :red
            )
            print_box(full_prompt, "White-Box Prompt", color=crayon"red bold", highlighters=high_prompt)
        end

        high_true = Highlighter(
            (data,i,j)->data[i,j] == true,
            foreground = :green
        )
        high_false = Highlighter(
            (data,i,j)->data[i,j] == false,
            foreground = :magenta
        )
        prob = exp(-nll)
        print_box(Any[sp.is_success loss r nll prob log_ppl], ["Success", "Loss", "Reward", "NLL", "Probability", "Log-perplexity"], color=crayon"yellow bold", highlighters=(high_true, high_false))
    end

    push!(mdp.data, (loss=loss, nll=nll, log_ppl=log_ppl, suffix=best_new_adv_suffix, is_success=sp.is_success, state=sp))

    cleanup()
    return (; sp, r)
end


POMDPs.discount(mdp::WhiteBoxMDP) = mdp.params.discount


"""
Clean up the CPU and GPU cache in Python and Julia.
"""
function cleanup()
    py"""
    gc.collect()
    """
    torch.cuda.empty_cache()
    GC.gc()
end
