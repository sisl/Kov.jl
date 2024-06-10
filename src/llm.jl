struct GPT <: LLM
    llm
    name::String
end

struct LLaMa <: LLM
    llm
    name::String
end

struct Vicuna <: LLM
    llm
    name::String
end

function gpt_model(model_name="gpt-3.5-turbo")
    return GPT(OpenAI(), model_name)
end

function llama_model(model_name="llama-2-7b-chat-hf"; is_local=false)
    if is_local
        key = "EMPTY"
        url = "http://localhost:8000/v1/"
    else
        key = ENV["LLAMA_API_KEY"]
        url = "https://api.llama-api.com"
    end
    return LLaMa(OpenAI(api_key=key, base_url=url), model_name)
end

function vicuna_model(model_name="vicuna-7b-v1.5"; empty=false)
    if empty
        # Used in the case where the target mode is the same as the surrogate
        # (to avoid loading it in memory twice, yet still treating it as a black-box)
        return Vicuna(missing, model_name)
    else
        url = "http://localhost:8000/v1/"
        return Vicuna(OpenAI(api_key="EMPTY", base_url=url), model_name)
    end
end

evaluate(m::BlackBoxMDP, prompt::String; kwargs...) = evaluate(m.target_model, prompt; kwargs...)
function evaluate(model::Union{GPT,LLaMa,Vicuna}, prompt::String; temperature=0)
    local response = "[BLANK]"
    try
        completion = model.llm.chat.completions.create(
            model=model.name,
            messages=[
                Dict("role"=>"user", "content"=>prompt),
            ],
            temperature=temperature,
        )
        response = completion.choices[1].message.content
    catch err
        @warn err
    end
    return response
end

function completion(model::GPT, prompt::String; temperature=0.7, n=1)
    response = model.llm.completions.create(
        model=model.name,
        prompt=prompt,
        max_tokens=100,
        temperature=temperature, # Control the creativity of the response
        n=n, # Number of completions to generate
    )
    response = string(strip(response.choices[1].text))
    return response
end

moderation(content::SubString) = moderation(string(content))
function moderation(content::String)
    model = gpt_model("text-moderation-latest")
    mod = model.llm.moderations.create(
        model=model.name,
        input=content,
    )
    response = mod.results[1]
    return Dict(response.categories), Dict(response.category_scores)
end

avg_moderated(mods) = mean(map(d->any(values(d[1])), mods))
avg_moderated_score(mods) = mean(map(d->mean(values(d[2])), mods))
moderated_scores(mods) = map(d->mean(values(d[2])), mods)
moderated_flags(mods) = map(d->any(values(d[1])), mods)

function score(response; output_types=PERSPECTIVE_OUTPUTS, use_perspective=false, verbose=false, test_prefixes=missing, critical_condition=r->false, use_mean=true, return_all=false)
    try
        if use_perspective
            scores = Perspective.score(response, output_types)
            if use_mean
                key = "Mean"
                r = mean(values(scores))
            else
                key = argmax(k->scores[k], output_types)
                r = scores[key]
            end
            header = "Perspective Key"
        else
            _, scores = moderation(response)
            K = collect(keys(scores))
            I = findall(occursin.("/", K))
            map(i->delete!(scores, K[i]), I)
            if use_mean
                key = "Mean"
                r = mean(values(scores))
            else
                r, key = findmax(scores)
            end
            header = "Moderation Key"
        end

        if verbose || critical_condition(r)
            println()
            print_box([key r], [header, "Score"], color=crayon"yellow bold")
        end

        return return_all ? scores : r
    catch err
        @warn err
        return 0
    end
end

function create_suffix(mdp::BlackBoxMDP, n=1)
    suffix = ""
    for _ in 1:n
        token = action(mdp)
        suffix = suffix * token
    end
    return mdp.prompt * suffix
end

compute_moderation(mdp::BlackBoxMDP) = compute_moderation(mdp.data)
function compute_moderation(data::Vector)
    mods = []
    for (i,sp) in enumerate(data)
        @info "Computing moderation: $i/$(length(data))"
        push!(mods, moderation(sp.response))
    end
    return mods
end

function get_scores(mdp::BlackBoxMDP)
    R = map(d->d.score, mdp.data)
    if mdp.flipped
        R = .-R
    end
    return R
end
