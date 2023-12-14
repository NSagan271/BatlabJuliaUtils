using DSP;

"""
    STFTwithdefaults(y::AbstractArray, nfft=256,
        noverlap=Int64(round(nfft/2)), window=hamming(nfft), zero_pad=true)
                                                        -> Matrix{ComplexF64}

Wrapper around `DSP.Periodograms.stft`, which takes the short-time Fourier
Transform (STFT) of a signal, with some reasonable default values set.

Inputs:
- `y`: one-dimensional signal of which to take the STFT.
- `nfft` (default: 256): length of each window of the STFT.
- `noverlap` (default: `nfft/2`): overlap between adjacent STFT windows.
- `window` (default: Hamming): function multiplicatively applied to each
    window to reduce spectral leakage.
- `zero_pad` (default: `true`): if set to `true`, zero-pad the beginning and end
    of y with (`nfft-1`) zeros on either end.

Output:
- `Sy`: STFT of y. Matrix with (`nfft / 2 + 1`) rows, each corresponding to a
    different frequency and `N` columns, where `N` is the number of time
    windows taken.
"""
function STFTwithdefaults(y::AbstractArray; nfft=256, noverlap=Int64(round(nfft/2)), window=hamming(nfft), zero_pad=true) :: Matrix{ComplexF64}
    if zero_pad
        y = vcat(zeros(nfft-1), y, zeros(nfft-1));
    end
    return stft(y, nfft, noverlap, window=window);
end

"""
    plotSTFTdb(Sy_db::Matrix; nfft=Int(2*floor(size(Sy_db, 1))), 
        noverlap=Int64(round(nfft/2)), crange=50, fs=250 kHz,
        plotting_kwargs...)

Plots spectrogram `Sy_db`, where `Sy_db` is in decibels (If `Sy` is the STFT of
signal `y`, then `Sy_db` is 20 times `log10` of the magnitude squared of `Sy`).

Inputs:
- `Sy_db`: STFT, in decibels.
- `nfft` (default: `2*(height of Sy_db - 1)`): length of each window of the STFT. 
    Used for axis labeling.
- `noverlap` (default: `nfft/2`): overlap between adjacent STFT windows. Used for
    axis labeling.
- `window` (default: Hamming): function multiplicatively applied to each
    window to reduce spectral leakage.
- `crange` (default: 50): the lowest value shown on the colorbar is the maximum
    value of `Sy_db`, minus crange. All elements of `Sy_db` that are smaller than
    this will show up in the spectrogram as this lowest value.
- `fs`: sampling frequency of the audio data, in Hertz. Default set in 
    Defaults.jl.
- `plotting_kwargs`: extra keyword arguments for plotting (passed into the
    heatmap function).
"""
function plotSTFTdb(Sy_db::Matrix; nfft=Int(2*(size(Sy_db, 1) - 1)), noverlap=Int64(round(nfft/2)), crange=50, fs=FS, plotting_kwargs...)
    stride = nfft-noverlap;
    kwargs = getplottingsettings("Milliseconds", "kHz", "Spectrogram"; plotting_kwargs...)
    replace!(Sy_db, NaN=>-10000);
    replace!(Sy_db, -Inf=>-10000);

    if isnothing(crange)
        crange = maximum(Sy_db) - minimum(Sy_db)
    end
    return heatmap(
            audioindextoms.((0:size(Sy_db, 2)-1) .* stride .+ 1),
            fftindextofrequency.(1:size(Sy_db, 1), nfft) ./ 1000,
            Sy_db,
            clims=(max(maximum(Sy_db) - crange, minimum(Sy_db)), maximum(Sy_db));
            kwargs...
        );
end

"""
    plotSTFT(Sy::Matrix{ComplexF64}; nfft=Int(2*floor(size(Sy_db, 1))), 
        noverlap=Int64(round(nfft/2)), crange=50, fs=250 kHz,
        plotting_kwargs...)

Plots the spectrogram corresponding to STFT `Sy` (i.e., plots a heatmap of
`Sy_db = 20*log10(|Sy|^2))`.

Inputs:
- `Sy`: STFT, in decibels.
- See `plotSTFTdb` for the rest of the arguments.
"""
function plotSTFT(Sy::Matrix{ComplexF64}; nfft=Int(2*(size(Sy_db, 1) - 1)), noverlap=Int64(round(nfft/2)), crange=50, fs=FS, kwargs...)
    return plotSTFTdb(pow2db.(abs.(Sy) .^ 2); nfft=nfft, noverlap=noverlap, fs=fs, crange=crange, kwargs...)
end

"""
    plotSTFTtime(y::AbstractArray; nfft=256, noverlap=Int64(round(nfft/2)),
        window=hamming(nfft), zero_pad=true, crange=50, fs=250 kHz,
        plotting_kwargs...)

Takes the STFT of `y` and then plots the spectrogram using `plotSTFT`.

Inputs:
- `y`: time-domain signal.
- `nfft`, `noverlap`, `window`, `zero_pad`: see `STFTwithdefaults`.
- `crange`, `fs`, `plotting_kwargs`: see `plotSTFTdb`
"""
function plotSTFTtime(y::AbstractArray; nfft=256, noverlap=Int64(round(nfft/2)),
        window=hamming(nfft), zero_pad=true, crange=50, fs=FS, kwargs...)
    return plotSTFT(STFTwithdefaults(y, nfft=nfft, noverlap=noverlap, window=window, zero_pad=zero_pad), nfft=nfft, noverlap=noverlap, fs=fs; kwargs...);
end