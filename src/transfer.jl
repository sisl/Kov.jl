function transfer(prompt, suffixes, target_model; temperature=0, verbose=true, kwargs...)
    responses = []
    scores = []
    mods = []
    for (i,suffix) in enumerate(suffixes)
        @info "Suffix $i/$(length(suffixes))"
        full_prompt = prompt * " " * suffix

        response = evaluate(target_model, full_prompt; temperature)
        push!(responses, response)

        r = score(response; verbose=verbose, kwargs...)
        push!(scores, r)

        # TODO. print_box
        println("—"^20, "[BlackBox]", "—"^20)
        printstyled("[PROMPT]: $full_prompt\n", color=:red)
        printstyled("[RESPONSE]: $response\n", color=:blue)
        printstyled("[SCORE]: $r\n", color=:yellow)
        println("—"^50)

        mod = moderation(response)
        push!(mods, mod)
    end
    return responses, scores, mods
end

"""
Take the suffixes from a training procedure and test the transfer to a target model.

e.g.,
    fn_training_data = i->"gpt3-kov-value-mgcg-advbench\$i-adv-mdp-data.bson"
"""
function test_transfer(fn_training_data::Function, target_model; is_baseline=false, n=1, benchmark_indices=1:5, temperature=0, kwargs...)
    results = Dict()
    for benchmark_idx in benchmark_indices
        @info "Testing transfer on AdvBench index $benchmark_idx"
        if is_baseline
            prompt = load_prompts(benchmark_idx)
            suffixes = fill("", n)
        else
            data = BSON.load(fn_training_data(benchmark_idx))[:data]
            prompt = data[1].prompt
            suffix = extract_suffix(fn_training_data, benchmark_idx)
            suffixes = fill(suffix, n)
        end
        responses, scores, mods = transfer(prompt, suffixes, target_model; temperature, kwargs...)
        results[benchmark_idx] = Dict(
            :prompt=>prompt,
            :suffixes=>suffixes,
            :responses=>responses,
            :scores=>scores,
            :mods=>mods,
        )
    end
    return results
end

function extract_suffix(fn_training_data, benchmark_idx)
    data = BSON.load(fn_training_data(benchmark_idx))[:data]
    idx = argmax(map(d->d.score, data))
    return data[idx].suffix
end

function extract_suffixes(fn_training_data::Function; benchmark_indices=1:5)
    prompts = Dict()
    for benchmark_idx in benchmark_indices
        prompts[benchmark_idx] = extract_suffix(fn_training_data, benchmark_idx)
    end
    return prompts
end
