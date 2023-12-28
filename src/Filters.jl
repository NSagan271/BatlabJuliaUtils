using DataStructures;
include("Defaults.jl");

"""
    movingaverage(x::AbstractArray, half_len::Int, stride=1)
                                                    -> Matrix{Real}

Applies a symmetrical moving average filter of width `2*half_len + 1` to signal
`x`. To make the output the same length as the input and avoid edge effects,
the input is padded by duplicating the first and last elements each `half_len`
times. The output is aligned with the input: i.e., if there is a large enough
local maximum at index `i` of the input, there will also be a local maximum at
the same index of the output.

`x` may have multiple columns, where each column represents a different set of
time-series data. In that case, the filter is applied separately to each
column.

Inputs:
- `x`: input vector, or matrix where each column is a different dataset (i.e.,
    channel).
- `half_len`: determines the length of the filter, as described above.
- `stride` (default: 1): if `stride` is not 1, then the output is downsampled
    by a factor of `stride`.

Output:
- `y`: result of applying a moving average filter to `x`.
"""
function movingaveragefilter(x::AbstractArray, half_len::Int, stride=1) :: Matrix{Real}
    x = vectortomatrix(x);
    y = zeros(length(1:size(x, 1)), size(x, 2))
    for i=1:size(x, 2)
        len=2*half_len + 1;
        xi_pad = vcat(repeat([x[1, i]], half_len), x[:, i], repeat([x[end, i]], (half_len+1)));
        y[:, i] = DSP.conv(xi_pad, ones(len)/len)[2*half_len+1:size(x, 1)+2*half_len];
    end
    return y; 
end

"""
    maxfilter(x::AbstractArray, half_len::Int) -> Matrix{Real}

Applies a maximum filter to the input: 
`y[n] = maximum(x[n-half_len:n+half_len])`.

Input signal `x` may have multiple columns; each column represents a different
set of time-series data.

Inputs:
- `x`: input vector, or matrix where each column is a different dataset (i.e.,
    channel).
- `half_len`: determines the length of the filter, as described above.

Output:
- `y`: output of the filter, as described above.
"""
function maxfilter(x::AbstractArray, half_len::Int) :: Matrix
    x = vectortomatrix(x);
    y = zeros(size(x));
    if length(x) == 0
        return x;
    end
    h = MutableBinaryMaxHeap{typeof(x[1])}();

    len = 2*half_len + 1;

    for k=1:size(x, 2)
        heap_handles = zeros(Int, len);
        extract_all!(h);
        for i=1:(size(x, 1)+half_len)
            if i > len
                delete!(h, heap_handles[(i % len) + 1]);
            end
            if i <= size(x, 1)
                heap_handles[(i % len) + 1] = push!(h, x[i, k]);
            end
            if i > half_len
                y[i - half_len, k] = top_with_handle(h)[1];
            end
        end
    end
    return y;
end

"""
    bandpassfilter(x::AbstractArray, min_Hz::Number, max_Hz::Number;
        fs=250 kHz) -> AbstractArray{Real}

Applies an ideal bandpass filter with cutoffs `min_Hz` and `max_Hz` to input
signal `x`.

Input signal `x` may have multiple columns; each column represents a different
set of time-series data

Inputs:
- `x`: input vector, or matrix where each column is a different dataset (i.e.,
    channel).
- `min_Hz`: lower cutoff of the filter.
- `max_Hz`: upper cutoff of the filter.
- `fs`: Sampling frequency, in Hertz. Default set in Defaults.jl.

Output:
- `y`: `x`, with frequencies below `min_Hz` or above `max_Hz` zeroed out.
"""
function bandpassfilter(x::AbstractArray, min_Hz::Number, max_Hz::Number;
        fs=FS) :: Matrix
    x_fft = colwisefft(x);
    y = colwiseifft(bandpassfilterFFT(x_fft, min_Hz, max_Hz; fs=fs));
    @assert all(abs.(imag.(y)) .< 1e-10);

    return real.(y);
end

"""
    bandpassfilterFFT(x_fft::AbstractArray, min_Hz::Number, max_Hz::Number;
        fs=250 kHz) -> AbstractArray

Same as `bandpassfilter`, except the input and output are in the frequency
domain.

Inputs:
- `x_fft`: input vector (or matrix) in the frequency domain.
- `min_Hz`: lower cutoff of the filter.
- `max_Hz`: upper cutoff of the filter.
- `fs`: Sampling frequency, in Hertz. Default set in Defaults.jl.

Output:
- `y_fft`: `x_fft`, with frequencies below `min_Hz` or above `max_Hz` zeroed
    out.
"""
function bandpassfilterFFT(x_fft::AbstractArray, min_Hz::Number, max_Hz::Number; fs=FS) :: AbstractArray
    y_fft = copy(x_fft);
    w0 = 2pi/size(x_fft, 1);
    min_omega = min_Hz / fs * 2pi;
    max_omega = max_Hz / fs * 2pi;

    min_idx = Int(floor(min_omega / w0)) + 1;
    max_idx = Int(ceil(max_omega / w0));

    y_fft[1:min_idx, :] .= 0;
    y_fft[end-min_idx+2:end, :] .= 0;
    y_fft[max_idx+1:end-max_idx+1, :] .= 0;

    return y_fft;
end

"""
    bandpassfilterspecgram(Sx::Matrix, min_Hz::Number, max_Hz::Number;
        nfft=2*(size(Sx, 1)-1), fs=FS) -> AbstractArray

Same as `bandpassfilter`, except the input and output are spectrograms.

Inputs:
- `Sx`: input spectrogram.
- `min_Hz`: lower cutoff of the filter.
- `max_Hz`: upper cutoff of the filter.
- `nfft` (default: `2*(height of Sx-1)`): size of the window used to produce
    the spectrogram.
- `fs`: Sampling frequency, in Hertz. Default set in Defaults.jl.

Output:
- `Sy`: `Sx`, with frequencies below `min_Hz` or above `max_Hz` zeroed out.
"""
function bandpassfilterspecgram(Sx::Matrix, min_Hz::Number, 
        max_Hz::Number; nfft=2*(size(Sx, 1)-1), fs=FS) :: AbstractArray
    Sy = Sx .* 0;

    w0 = 2pi/nfft;
    min_omega = min_Hz / fs * 2pi;
    max_omega = max_Hz / fs * 2pi;

    min_idx = Int(floor(min_omega / w0)) + 1;
    max_idx = Int(ceil(max_omega / w0));

    Sy[min_idx+1:max_idx, :] = Sx[min_idx+1:max_idx, :];
    return Sy
end

"""
    circconv(x::AbstractArray, h::AbstractArray; real_output=true) -> AbstractArray

Perform circular convolution in the frequency domain:
`y = iFFT(FFT(x) * FFT(h))`.

Inputs:
- `x`, `h`: two vectors of the same dimension, or matrtices where each column
    is a different channel of data.
- `real_output` (default: `true`): whether to take the real component of the
    output before returning (`false` to leave the output a complex number).

Output:
- circular convolution of x and y
"""
function circconv(x::AbstractArray, h::AbstractArray; real_output=true) :: AbstractArray
    x = reshape(x, (size(x, 1), :));
    h = reshape(h, (size(h, 1), :));
    y = colwiseifft(colwisefft(x) .* colwisefft(h));
    if real_output
        return real.(y);
    end
    return y;
end

"""
    deconvolve(Y::AbstractArray, X::AbstractArray; fft_thresh=0.1) 
                                                    -> AbstractArray

Perform deconvolution: i.e., given linear time-invariant system `Y` and input
`X`, such that `Y` is `X` convolved with impulse response `H`, find `H` using
Fourier transforms (`H = iFFT(FFT(Y) ./ FFT(X))`).

Optionally, zero out indices of `FFT(Y) ./ FFT(X)` where the magnitude of 
`FFT(X)` is above some threshold, `fft_thresh`.

`X` and `Y` must have the same dimensions. If they are matrices, deconvolution
is applied separately to each column.

Inputs:
- `Y`: system output; either a vector or a matrix where each column is a
    different data channel.
- `X`: system input; same dimensions as `Y`.
- `fft_thresh` (default: 0): described above.

Outputs:
- `H`: estimated impulse response; same dimensions as `X` and `Y`.
"""
function deconvolve(Y::AbstractArray, X::AbstractArray; fft_thresh=0) :: AbstractArray
    X_fft = colwisefft(X);
    return real.(colwiseifft(colwisefft(Y) ./ X_fft .* (abs.(X_fft) .>= fft_thresh)));
end