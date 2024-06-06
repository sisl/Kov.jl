function run_baseline(mdp::BlackBoxMDP, n, suffix="")
    R = []
    data = []
    for i in 1:n
        @info "Baseline: $i/$n"
        local s0 = rand(initialstate(mdp))
        local sp, r = @gen(:sp, :r)(mdp, s0, suffix)
        push!(R, r)
        push!(data, sp)
    end
    return R, data
end

function sample_tokens(tokenizer, n; not_allowed_tokens=missing, device="cuda")
    token_ids = torch.arange(tokenizer.vocab_size, device=device)
    if ismissing(not_allowed_tokens)
        filtered_tokens = token_ids
    else
        filtered_tokens = py"filter_forbidden_toks"(token_ids, not_allowed_tokens)
    end
    tokens = torch.randint(0, filtered_tokens.shape[1], (n,1), device=device).squeeze(1)    
    return tokens
end
