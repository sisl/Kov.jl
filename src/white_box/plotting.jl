function plot_loss(losses; save=true, filename="losses.png", flipped=false)
    default(fontfamily="Computer Modern", framestyle=:box)
    kwargs = (xlabel="epoch", ylabel="loss", lw=2, c=:crimson, label=false)
    losses = flipped ? .-losses : losses
    if length(losses) == 1
        scatter([1], [losses]; kwargs...)
    else
        plot(losses; kwargs...)
    end
    save && savefig(filename)
    plot!()
end

MAX_COLUMN_WIDTH = 120
print_box(text::Union{Real}, header::Union{String,SubString}; kwargs...) = print_box([text], [header]; alignment=:r, kwargs...)
print_box(text::Union{String,SubString}, header::Union{String,SubString}; kwargs...) = print_box([text], [header]; columns_width=min(MAX_COLUMN_WIDTH, displaysize(stdout)[2]-8), kwargs...)
function print_box(data, headers::Vector; color=crayon"yellow bold", kwargs...)
    pretty_table(data;
        header=headers,
        alignment=:l,
        header_crayon=color,
        autowrap=true,
        linebreaks=true,
        kwargs...)
end

print_llm(prompt, response::SubString; kwargs...) = print_llm(prompt, string(response); kwargs...)
function print_llm(prompt::String, response::String; prefix="", response_color=:blue, kwargs...)
    if !isempty(prefix)
        prefix = prefix * " "
    end
    pretty_table(["$(prefix)Prompt" prompt; "$(prefix)Response" response];
        body_hlines=[1],
        show_header=false,
        autowrap=true,
        linebreaks=true,
        crop=:none,
        alignment=:l,
        columns_width=[10,min(MAX_COLUMN_WIDTH, displaysize(stdout)[2]-8)],
        highlighters=(
            Highlighter(
                (data,i,j)->i==1 && j==1,
                foreground=:red,
                bold=true
            ),
            Highlighter(
                (data,i,j)->i==1 && j != 1,
                foreground=:red
            ),
            Highlighter(
                (data,i,j)->i==2 && j==1,
                foreground=response_color,
                bold=true
            ),
            Highlighter(
                (data,i,j)->i==2 && j != 1,
                foreground=response_color
            ),
        ),
        kwargs...
    )
end
