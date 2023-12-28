using Statistics;
using DSP;

"""
    getnoisesampleidxs(mic_data::AbstractArray; window_size=200)
                                                    -> UnitRange{Int}

Given audio data, find the longest segment that is just noise.

First, the algorithm divides the data into segments of length `window_size`. The
amplitude of a window is defined as the maximum deviation of `mic_data` from its
mean (over the window). The maximum noise level is set at twice the minimum
amplitude of any window. The algorithm finds the indices of the longest section
of `mic_data` where no window has an amplitude above the noise level.

Inputs:
- `mic_data`: matrix of audio data, where each column is a different channel.
- `window_size`: length of the windows described above.

Outputs:
- `UnitRange` (i.e., the datatype of the object `1:10`) of the indices of the
    longest segment of the data that is only noise.
"""
function getnoisesampleidxs(mic_data::AbstractArray; window_size=200) :: UnitRange{Int}
    # If mic_data is one-dimensional, make it a matrix with a single column.
    # Otherwise, it'll be unchanged.
    mic_data = vectortomatrix(mic_data);

    # do one pass to find the noise level
    min_amplitude = Inf;

    get_amplitude = (data_slice) -> maximum(abs.(data_slice .- mean(data_slice; dims=1)));
    for i=1:window_size:size(mic_data, 1)-window_size+1
        min_amplitude = min(min_amplitude, get_amplitude(mic_data[i:i+window_size-1, :]))
    end
    amp_thresh = min_amplitude * 2;

    sample_start = 1;
    best_noise_idxs = 1:1;
    
    for i=1:window_size:(size(mic_data, 1)-window_size)
        win_amplitude = get_amplitude(mic_data[i:i+window_size-1, :]);
        if win_amplitude > amp_thresh
            sample_end = i;
            # println(sample_end - sample_start + 1)
            if sample_end - sample_start + 1 > length(best_noise_idxs)
                best_noise_idxs = sample_start:sample_end;
            end
            sample_start = i+window_size;
        end
    end

    sample_end = size(mic_data, 1) - window_size;
    # println(sample_end - sample_start + 1)
    if sample_end - sample_start + 1 > length(best_noise_idxs)
        best_noise_idxs = sample_start:sample_end;
    end
    return best_noise_idxs;
end

"""
    windowedenergy(x::AbstractArray, window_size::Int; window=hamming(nfft))
                                                            -> Matrix{Real}

Finds the energy over sliding windows of the signal `x`, where the windows have
stride 1 (i.e., if the first window starts at index 1, the second window starts
at index 2, etc.)

Inputs:
- `x`: vector of data, or matrix where each column is a different channel of
    data.
- `window_size` (default: 64): window length, in samples.
- `window` (default: `ones(window_size)`): vector to multiply each window by.
    The default, `ones`, multiplies each element by 1. To emphasize the center
    of long windows, you can use windows like `hamming(window_size)` from the
    `DSP` package.
"""
function windowedenergy(x::AbstractArray, window_size::Int; window=ones(window_size)) :: Matrix
    BLOCK_SIZE=500_000;
    
    pad_len = Int(round(window_size / 2));

    # If x is one-dimensional, make it a matrix with a single column.
    # Otherwise, it'll be unchanged.
    x = vectortomatrix(x);
    retval = zeros(size(x));
    x = vcat(zeros(pad_len, size(x, 2)), x, zeros(window_size-1-pad_len, size(x, 2)));

    for k=1:size(x, 2)
        for start_idx=1:BLOCK_SIZE:size(x, 1)-window_size
            N = min(BLOCK_SIZE, size(retval, 1) - start_idx + 1);
            
            retval[start_idx:start_idx+N-1, k] = 
                    mapslices(X -> sum(abs.(X) .^ 2), stft(x[start_idx:start_idx+N+window_size-2, k], 
                              window_size, (window_size-1), window=window); dims=1);
        end
    end
    return retval;
end

"""
    estimatesnr(y::AbstractArray, noise_sample::AbstractArray, window_size=256,
        window=ones(window_size)) -> Matrix{Real}

Crude estimate of the signal-to-noise ratio of the input signal `y`. The signal
and noise levels are computed by taking the energy over sliding windows (with
stride 1) of `y` and `noise_sample`, respectively. The SNR is estimated as the
signal level of each window, divided by the mean noise level.

Inputs:
- `y`: input signal, which may be a vector, or a matrix where each column is a
    different channel.
- `noise_sample`: sample of noise, with the same number of channels as `y`.
- `window_size` (default: 64): window length, in samples.
- `window` (default: `ones(window_size)`): vector to multiply each window by.
    The default, `ones`, multiplies each element by 1. To emphasize the center
    of long windows, you can use windows like `hamming(window_size)` from the
    `DSP` package.

Output:
- `snr_est`: estimated SNR, in decibels (log scale, times 20) of every
    timepoint in `y`.
"""
function estimatesnr(y::AbstractArray, noise_sample::AbstractArray; window_size=64, window=ones(window_size)) :: Matrix
    WINDOW_LEN=500_000;

    # If y is one-dimensional, make it a matrix with a single column.
    # Otherwise, it'll be unchanged.
    y = vectortomatrix(y);
    noise_level = mean(windowedenergy(noise_sample, window_size, window=window); dims=1);

    snr_est = windowedenergy(y, window_size, window=window) ./ noise_level;
    return 20 .* log10.(max.(snr_est, 1e-6))
end