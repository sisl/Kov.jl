const ADVBENCH_FILENAME = joinpath(@__DIR__, "..", "..", "data", "advbench_subset.csv")

function load_advbench(filename=ADVBENCH_FILENAME)
    D = CSV.read(filename, DataFrame)
    prompts = D[!, :goal]
    targets = D[!, :target]
    return (; prompts, targets)
end

function load_prompts(idx=missing; filename=ADVBENCH_FILENAME)
    prompts = load_advbench(filename).prompts
    return ismissing(idx) ? prompts : prompts[idx]
end

function load_targets(idx=missing; filename=ADVBENCH_FILENAME)
    targets = load_advbench(filename).targets
    return ismissing(idx) ? targets : targets[idx]
end
