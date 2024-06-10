module LivePlots

using Reexport
@reexport using Plots
default(fontfamily="Computer Modern", framestyle=:box)

export liveplot, update!, clear!

global plt = missing

function liveplot(y=[]; kwargs...)
    if isempty(y)
        global plt = plot(; kwargs...)
    else
        global plt = plot(y; kwargs...)
    end
    return plt
end

function update!(y; show=false, kwargs...)
    global plt
    if ismissing(plt)
        Y = [y]
        global plt = plot(Y; kwargs...)
    else
        Y = plt[1][1][:y]
        push!(Y, y)
        global plt = plot(Y; plt.attr..., kwargs...)
    end
    show && display(plt)
    return plt
end

function clear!()
    global plt = missing
end

end # module LivePlots
