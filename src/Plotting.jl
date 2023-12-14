include("Defaults.jl");
using Plots;
"""
    getplottingsettings(xlabel::String, ylabel::String, title::String;
        kwargs...) -> Dict{Symbol, Any}

Builds a dictionary of keyword arguments to pass into any plotting function,
populated with the provided labels and titles, default font sizes and plot
dimensions, and any additional keyword arguments passed into the function.

Inputs:
- `xlabel`: label of the x-axis.
- `ylabel`: label of the y-axis.
- `title`: plot title.
- `kwargs...`: you can pass in any other keyword arguments to specify plotting
    parameters or override the defaults set here.

Output:
- `kwargs_with_defaults`: dictionary of plotting keyword arguments.
"""
function getplottingsettings(xlabel::String, ylabel::String, title::String; kwargs...) :: Dict{Symbol, Any}
    kwargs_with_defaults = Dict{Symbol, Any}(
        :size => DEFAULT_PLOT_DIM, 
        :titlefontsize => 12,
        :labelfontsize => 10,
        :legendfontsize => 8,
        :xtickfontsize => 8,
        :ytickfontsize => 8,
        :html_output_format => :png,
        :left_margin => 10Plots.mm,
        :bottom_margin => 10Plots.mm,
        :xlabel => xlabel,
        :ylabel => ylabel,
        :title => title
    );
    for key=keys(kwargs)
        kwargs_with_defaults[key] = kwargs[key];
    end
    return kwargs_with_defaults;
end

""" 
    myplot(args...; kwargs...)

Mimics Julia's `plot` function, except with defaults from `getplottingsettings`
passed in. You can pass in any arguments you would to the regular `plot`
function.
"""
function myplot(args...; kwargs...)
    return plot(args...; getplottingsettings("x", "y", "Title"; kwargs...)...);
end

""" 
    myplot!(args...; kwargs...)

Same as `myplot`, except adds to the last plot produced instead of making a new
plot.
"""
function myplot!(args...; kwargs...)
    return plot!(args...; getplottingsettings("x", "y", "Title"; kwargs...)...);
end

"""
    plotmicdata(idxs::AbstractArray{Int64}, y::AbstractArray; plot_idxs=idxs,
        kwargs...)

Plot a time segment of audio data `y`, with milliseconds on the x-axis and
voltage on the y-axis, and default plotting settings from
`getplottingsettings`.

Inputs:
- `idxs`: time indices of input audio data to plot.
- `y`: audio data, where each column is one microphone.
- `plot_idxs` (default: `idxs`): the x-axis labels of the plot will be
    `audioindextoms.(plot_idxs)`. Use this parameter if the x-axis labels you
    want don't match the value you passed in for `idxs`.
- `kwargs...`: you can pass in any additional keyword arguments to set plotting
    parameters.
"""
function plotmicdata(idxs::AbstractArray{Int64}, y::AbstractArray; plot_idxs=idxs, kwargs...)
    return plot(audioindextoms.(plot_idxs), Float64.(y[idxs, :]); 
                getplottingsettings("Milliseconds", "Voltage", "Mic Data"; kwargs...)...);
end

"""
    plotmicdata!(idxs::AbstractArray{Int64}, y::AbstractArray; plot_idxs=idxs,
        kwargs...)

Same as `plotmicdata(idxs, y)`, except adds to the last plot produced instead
of making a new plot.
"""
function plotmicdata!(idxs::AbstractArray{Int64}, y::AbstractArray; plot_idxs=idxs, kwargs...)
    return plot!(audioindextoms.(plot_idxs), Float64.(y[idxs, :]); 
                getplottingsettings("Milliseconds", "Voltage", "Mic Data"; kwargs...)...);
end

"""
    plotmicdata(y::AbstractArray; plot_idxs=1:size(y, 1), kwargs...)

Plot audio data, with milliseconds on the x-axis and voltage on the y-axis, and
default plotting settings from `getplottingsettings`. Plots the full length of
`y`.

Inputs:
- `y`: audio data, where each column is one microphone.
- `plot_idxs` (default: `1, 2, 3,...`): the x-axis labels of the plot will be
    `audioindextoms.(plot_idxs)`.
- `kwargs...`: you can pass in any additional keyword arguments to set plotting
    parameters.
"""
function plotmicdata(y::AbstractArray; plot_idxs=1:size(y, 1), kwargs...)
    return plotmicdata(1:size(y, 1), y, plot_idxs=plot_idxs; kwargs...)
end

"""
    plotmicdata!(y::AbstractArray; plot_idxs=1:size(y, 1), kwargs...)

Same as `plotmicdata(y)`, except adds to the last plot produced instead of
making a new plot.
"""
function plotmicdata!(y::AbstractArray; plot_idxs=1:size(y, 1), kwargs...)
    return plotmicdata!(1:size(y, 1), y, plot_idxs=plot_idxs; kwargs...)
end

"""
    plotfftmag(idxs::AbstractArray{Int64}, y::AbstractArray;
        fft_idxs=1:length(idxs), kwargs...)

Plot the Fourier transform magnitude of a time segment of real-valued
time-domain audio data `y`.

Inputs:
- `idxs`: time indices of input audio data to take the Fourier transform of.
- `y`: time-domain audio data, where each column is one microphone.
- `fft_indices` (default: plot all indices): indices of the Fourier transform
    to plot. Use this parameter if you only want to plot some frequencies.
- `kwargs...`: you can pass in any additional keyword arguments to set plotting
    parameters.
"""
function plotfftmag(idxs::AbstractArray{Int64}, y::AbstractArray; fft_idxs=1:length(idxs), kwargs...)
    N=length(idxs);
    return plot(getfftfrequencies(N)[fft_idxs] ./ 1000, abs.(colwisefft(y[idxs, :])[fft_idxs, :]);
                getplottingsettings("kHz", "Magnitude", "FFT Magnitude"; kwargs...)...);
end

"""
    plotfftmag(y::AbstractArray; fft_idxs=1:size(y, 1), kwargs...)

Plot the Fourier transform magnitude of real-value time-domain audio data `y`.
Takes the Fourier transform of the full length of `y`.

Inputs:
- `y`: time-domain audio data, where each column is one microphone.
- `fft_indices` (default: plot all indices): indices of the Fourier transform
    to plot. Use this parameter if you only want to plot some frequencies.
- `kwargs...`: you can pass in any additional keyword arguments to set plotting
    parameters.
"""
function plotfftmag(y::AbstractArray; fft_idxs=1:size(y, 1), kwargs...)
    return plotfftmag(1:size(y, 1), y; fft_idxs=fft_idxs, kwargs...);
end