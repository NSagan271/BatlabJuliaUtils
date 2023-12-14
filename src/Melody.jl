"""
    findmelody(chirp_seq_single_mic::ChirpSequence, peak_snr_thresh::Real;
        nfft=256, bandpass_filter=(20_000, 100_000), maximum_melody_slope=5)
                                                            -> Vector{Int64}

Given a chirp sequence (for a single microphone), estimate the "melody" (i.e.,
the fundamental harmonic) of the vocalization using the spectrogram.

The melody is traced as follows:
1. Find when the SNR goes over the threshold previously set for the peak SNR of
    chirp sequences, and find the melody of the chirp at that point (here, the
    SNR is hopefully high enough to accurately estimate the melody).
2. Work backwards until the beginning of the chirp, at each index looking for
    the strongest frequency within some small range of the last frequency
    found. This range is determined by the parameter `maximum_melody_slope`.
3. Repeat, but this time work towards the end of the chirp. To avoid picking up
    echos, enforce that, once the slope of the melody (with respect to time)
    becomes negative, it can never become positive.

After tracing the melody, we need to see if we found the fundamental harmonic
or some higher harmonic. This is done by taking the loudest part of the melody
and dividing the frequency by 2, 3, etc. until we go below 20 kHz. The fundamental harmonic is the lowest such frequency with power at most
`melody_drop_thresh_db` below the loudest harmonic.

The melody is computed in terms of Fourier transform indices. To find the 
melody in Hertz, use the `fftindextofrequency` function or use 
`findmelodyhertz` instead.

Inputs:
- `chirp_seq_single_mic`: input chirp sequence object (for a single microphone).
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `nfft` (default: 256): window size for computing the spectrogram.
- `bandpass_filter` (default: `(20_000, 100_000)`): before computing the melody,
    a band-pass filter is applied to the spectrogram. `bandpass_filter` is a
    tuple of the (lower cutoff, upper cutoff), in Hertz.
- `maximum_melody_slope` (default: 5): maximum amount, in Fourier transform 
    indices, that that the melody is allowed to change from one index to the 
    next.
- `melody_drop_thresh_db` (default: 20): used to find the fundamental harmonic; described above.

Output:
- `melody`: described above.
"""
function findmelody(chirp_seq_single_mic::ChirpSequence, peak_snr_thresh::Real;
        nfft=256, bandpass_filter=(20_000, 100_000), maximum_melody_slope=5,
        melody_drop_thresh_db=20) :: Vector{Int64}
    pad_len = Int64(round(nfft/2));
    x = vcat(zeros(pad_len), chirp_seq_single_mic.mic_data, zeros(nfft-1-pad_len));
    snr = chirp_seq_single_mic.snr_data;
    # Get the spectrogram
    Sx = stft(x, nfft, nfft-1, window=hamming(nfft));
    # bandpass filter
    Sx = bandpassfilterspecgram(Sx, bandpass_filter[1], bandpass_filter[2], nfft=nfft);
    Sx_db = pow2db.(abs.(Sx) .^ 2); # in decibels
    N = size(Sx, 2);
    Sx_height = size(Sx, 1);

    melody = ones(Int64, N);

    # Find the index where the SNR goes above peak_snr_thresh:
    idx1 = findfirst(snr .>= peak_snr_thresh)[1];
    melody[idx1] = argmax(Sx_db[:, idx1])[1];

    done = false;
    harmonic = 1;
    while !done
        melody[idx1] = Int64(round(melody[idx1]/harmonic));
        
        # trace the melody backwards to the beginning of the chirp
        for idx=idx1-1:-1:1
            prev_freq = melody[idx+1];
            band_freq_idxs = max(prev_freq-maximum_melody_slope, 1):min(prev_freq+maximum_melody_slope, Sx_height);
            band = Sx_db[band_freq_idxs, idx];
            melody[idx] = argmax(band) + band_freq_idxs[1] - 1;
        end
        # trace the melody forward to the end of the chirp, enforcing that, once the
        # slope of the melody becomes negative, it can never become positive.
        downward_trajectory = false;
        for idx=idx1+1:N
            prev_freq = melody[idx-1];
            band_freq_idxs = max(prev_freq-maximum_melody_slope, 1):min(prev_freq+maximum_melody_slope, Sx_height);
            if downward_trajectory
                band_freq_idxs = max(prev_freq-maximum_melody_slope, 1):prev_freq;
            end
            band = Sx_db[band_freq_idxs, idx];
            melody[idx] = argmax(band) + band_freq_idxs[1] - 1;
    
            if melody[idx] < melody[idx-1]
                downward_trajectory = true;
            end
        end

        # find the fundamental frequency
        if harmonic == 1
            melody_db = [Sx_db[melody[i], i] for i=1:N];
            loudest_idx = argmax(melody_db);

            test_harmonic = 1;
            
            while Int64(round(melody[loudest_idx]/test_harmonic)) > bandpass_filter[1] / FS * nfft
                if Sx_db[Int64(round(melody[loudest_idx]/test_harmonic)), loudest_idx] > 
                            Sx_db[melody[loudest_idx], loudest_idx] - melody_drop_thresh_db
                    harmonic = test_harmonic;
                end
                test_harmonic += 1;
            end
            if harmonic == 1
                done = true;
            end
        else
            done = true;
        end
    end
    return melody;
end

"""
    findmelodyhertz(chirp_seq_single_mic::ChirpSequence, peak_snr_thresh::Real;
        nfft=256, bandpass_filter=(20_000, 100_000), maximum_melody_slope=5,
        fs=250_000) -> Vector{Float64}

Same as `findmelody`, but converts the result to Hertz.
"""
function findmelodyhertz(chirp_seq_single_mic::ChirpSequence, peak_snr_thresh::Real;
        nfft=256, bandpass_filter=(20_000, 100_000), maximum_melody_slope=5,
        melody_drop_thresh_db=20, fs=FS) :: Vector{Float64}
    return fftindextofrequency.(findmelody(chirp_seq_single_mic, peak_snr_thresh;
        nfft=nfft, bandpass_filter=bandpass_filter, maximum_melody_slope=maximum_melody_slope),
        melody_drop_thresh_db=melody_drop_thresh_db,
        nfft, fs=fs);
end

"""
    getharmonic(chirp_seq_single_mic::ChirpSequence, melody::Vector{Int64},
        harmonic_num::Int64; nfft=256, band_size=2) -> Vector{Int64}

Given the output of `findmelody` for `chirp_seq_single_mic`, find the harmonic
given by `harmonic_num`. This is done by searching for the strongest
frequencies in a region of radius `band_size` around `harmonic_num` times the
melody.

Inputs:
- `chirp_seq_single_mic`: input chirp sequence object (for a single microphone).
- `melody`: output of `findmelody`.
- `harmonic_num`: which harmonic to find (e.g., 2, 3, etc.)
- `nfft` (default: 256): window size for computing the spectrogram.
- `band_size` (default: 2): described above.

Outputs:
- `harmonic`: array that contains the estimated harmonic, in Fourier transform
    indices.
"""
function getharmonic(chirp_seq_single_mic::ChirpSequence, melody::Vector{Int64},
        harmonic_num::Int64; nfft=256, band_size=3)
    pad_len = Int64(round(nfft/2));
    x = vcat(zeros(pad_len), chirp_seq_single_mic.mic_data, zeros(nfft-1-pad_len));
    snr = chirp_seq_single_mic.snr_data;
    # Get the spectrogram
    Sx = stft(x, nfft, nfft-1, window=hamming(nfft));
    Sx_db = pow2db.(abs.(Sx) .^ 2); # in decibels

    N = size(Sx, 2);
    Sx_height = size(Sx, 1);

    harmonic = min.(melody*harmonic_num, Sx_height);
    for (i, note)=enumerate(melody)
        band_start = harmonic_num*note-band_size*harmonic_num;
        freq_band = band_start:min(harmonic_num*note+band_size*harmonic_num, Sx_height);
        if length(freq_band) > 0
            harmonic_val = argmax(Sx_db[freq_band, i]) + band_start-1;
            if Sx_db[harmonic_val, i] > Sx_db[harmonic[i], i] + 1
                harmonic[i] = harmonic_val;
            end
        else
            harmonic[i] = Sx_height;
        end
    end
    return harmonic;
end

"""
        estimatechirpbounds(chirp_seq_single_mic::ChirpSequence,
        melody::Vector{Int64}, peak_snr_thresh::Real; nfft=256,
        bandpass_filter=(20_000, 100_000),melody_drop_thresh_db=20, melody_thresh_db_low=-20,moving_avg_size=10) -> Int64

Given a chirp sequence object (from a single mic) and the melody estimated by
`findmelody`, estimate the end index of the chirp (i.e., separate the chirp
from the echos) as follows:

1. Find the point where the melody is the strongest.
2. Find the first index, after this point, where the melody strength drops over
    `melody_drop_thresh_db` decibels from its peak value (if this cutoff value
    is below `melody_thresh_db_low`, we instead find where the melody strength
    goes below `melody_thresh_db_low`).
    a. If the melody strength never drops below this threshold, then just return
        the last index of the chirp sequence.
3. Apply a moving average filter to the melody strength.
4. Apply the following heuristic:
    - The end of the chirp is the first local minimum of the melody strength
    after the index from step 2, or the first time the melody strength dips
    below `melody_thresh_db_low`, whichever comes first.
    - If neither event happens, return the last index of the chirp sequence.

Inputs:
- `chirp_seq_single_mic`: input chirp sequence object (for a single microphone).
- `melody`: result of `findmelody`.
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `nfft` (default: 256): window size for computing the spectrogram.
- `bandpass_filter` (default: `(20_000, 100_000)`): before computing the melody,
    a band-pass filter is applied to the spectrogram. `bandpass_filter` is a
    tuple of the (lower cutoff, upper cutoff), in Hertz.
- `melody_drop_thresh_db`, `melody_thresh_db_low` (defaults: 20, -20):
    described in step 2 above.
- `melody_thresh_db_start`: the start index of the chirp is computed as the
    first index where the melody passes this threshold. It usually works best
    when this is a bit above the `melody_thresh_db_low`.
- `moving_avg_size`: radius of the moving average filter from step 4 above.

Output:
- `chirp_start_index`: estimated start index of the chirp, in indices since the
    start of the chirp sequence.
- `chirp_end_index`: estimated end index of the chirp, in indices since the
    start of the chirp sequence.
"""
function estimatechirpbounds(chirp_seq_single_mic::ChirpSequence,
        melody::Vector{Int64}, peak_snr_thresh::Real;
        nfft=256, bandpass_filter=(20_000, 100_000),
        melody_drop_thresh_db=20, melody_thresh_db_low=-20, 
        melody_thresh_db_start=-10, moving_avg_size=10)

    pad_len = Int64(round(nfft/2));
    x = vcat(zeros(pad_len), chirp_seq_single_mic.mic_data, zeros(nfft-1-pad_len));
    # Get the spectrogram
    Sx = stft(x, nfft, nfft-1, window=hamming(nfft));
    # bandpass filter
    Sx = bandpassfilterspecgram(Sx, bandpass_filter[1], bandpass_filter[2], nfft=nfft);
    Sx_db = pow2db.(abs.(Sx) .^ 2); # in decibels

    melody_thresh_db = max(melody_thresh_db_low, maximum(Sx_db) - melody_drop_thresh_db);

    N = length(melody);

    melody_db = [Sx_db[melody[i], i] for i=1:N];
    chirp_start = findfirst(melody_db .>= min(melody_thresh_db_start, melody_thresh_db));

    # The end of the chirp must occur after the snr reaches the peak threshold
    idx1 = argmax(melody_db);
    cutoff = findfirst(melody_db[idx1:end] .< melody_thresh_db);
    if isnothing(cutoff)
        return N;
    end
    cutoff = cutoff[1] + idx1 - 1;

    # Heuristic: the end of the chirp is the first local minimum of a moving
    # average-filtered melody_db after `cutoff`, or the first time the melody_db
    # dips below `melody_thresh_db_low`, whichever comes first. If neither event happens,
    # return `N`.
    averaged_melody_db = movingaveragefilter(melody_db[cutoff:end], moving_avg_size);
    is_local_min = (averaged_melody_db[2:end-1] .<= averaged_melody_db[1:end-2]) .&
                        (averaged_melody_db[2:end-1] .<= averaged_melody_db[3:end]);
    first_local_min = findfirst(is_local_min);
    first_local_min = isnothing(first_local_min) ? N-cutoff : first_local_min[1];

    first_below_thresh = findfirst(melody_db[cutoff:end-1] .<= melody_thresh_db_low);
    first_below_thresh = isnothing(first_below_thresh) ? N-cutoff : first_below_thresh[1];

    return chirp_start, min(first_local_min, first_below_thresh)  + cutoff - 1;
end

"""
    getchirpstartandendindices(
        chirp_sequence_all_mics::Dict{Int64, ChirpSequence}, peak_snr_thresh::Real;
        chirp_kwargs...) -> Dict{Int64, Int64}, Dict{Int64, Int64}

Given a dictionary mapping microphones to `ChirpSequence` objects, run
`findmelody` and `estimatechirpbounds` for each `ChirpSequence` and return two
dictionaries mapping microphones to chirp start indices and chirp end indices,
respectively. Indices are calculated in samples since the start of the chirp
sequence (i.e., since the beginning of the `mic_data` array of the
`ChirpSequence` object).

Inputs:
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `chirp_kwargs`: you can additionally pass in any keyword arguments for
`findmelody` and/or `estimate_chirp_bounds`.

Outputs:
- `chirp_starts`: dictionary mapping microphone indices to their respective
    chirp start indices (in samples since the start of the chirp sequence).
- `chirp_ends`: dictionary mapping microphone indices to their respective
    chirp end indices (in samples since the start of the chirp sequence).
"""
function getchirpstartandendindices(chirp_sequence_all_mics::Dict{Int64, ChirpSequence}, peak_snr_thresh::Real; chirp_kwargs...)
    melody_kwargs, chirp_bound_kwargs, _ = separatechirpkwargs(;chirp_kwargs...);
    chirp_ends = Dict{Int64, Int64}();
    chirp_starts = Dict{Int64, Int64}();

    for mic_idx = keys(chirp_sequence_all_mics)
        melody = findmelody(chirp_sequence_all_mics[mic_idx], peak_snr_thresh;  melody_kwargs...);
        chirp_starts[mic_idx], chirp_ends[mic_idx] = estimatechirpbounds(chirp_sequence_all_mics[mic_idx],
                melody, peak_snr_thresh; chirp_bound_kwargs...);
    end
    return chirp_starts, chirp_ends;
end

"""
    plotmelody(chirp_seq_single_mic::ChirpSequence, melody::Vector{Int64},
        chirp_end=nothing; nfft=256, bandpass_filter=(20_000, 100_000),
        melody_color="blue", end_color="cyan")

Plots the melody estimated by `findmelody`, overlayed on the spectrogram of
the chirp sequence. Optionally, plot a vertical line at the estimated end of
the chirp sequence.

Inputs:
- `chirp_seq_single_mic`: input chirp sequence object (for a single microphone).
- `melody`: result of `findmelody`.
- `chirp_end` (default: `nothing`): optionally, result of `estimatechirpend`.
    If `chirp_end` is nothing, then no vertical line is plotted.
- `nfft` (default: 256): window size for computing the spectrogram.
- `bandpass_filter` (default: `(20_000, 100_000)`): before computing the melody,
    a band-pass filter is applied to the spectrogram. `bandpass_filter` is a
    tuple of the (lower cutoff, upper cutoff), in Hertz.
- melody_color (default: "blue"): color used to plot the melody.
- end_color (default: "cyan"): color used to plot the vertical line at the end
    of the chirp sequence.
"""
function plotmelody(chirp_seq_single_mic::ChirpSequence, melody::Vector{Int64}, chirp_bounds=nothing;
        nfft=256, bandpass_filter=(20_000, 100_000), melody_color="blue", end_color=1)
    pad_len = Int64(round(nfft/2));
    x = vcat(zeros(pad_len), chirp_seq_single_mic.mic_data, zeros(nfft-1-pad_len));
    Sx = stft(x, nfft, nfft-1, window=hamming(nfft));
    Sx = bandpassfilterspecgram(Sx, bandpass_filter[1], bandpass_filter[2], nfft=nfft);
    N = size(Sx, 2);
    
    p = plotSTFT(Sx, nfft=nfft, noverlap=nfft-1)
    plot!(audioindextoms.(1:N), fftindextofrequency.(melody, nfft) / 1000, linewidth=3, color=melody_color, label="Melody")
    if !isnothing(chirp_bounds)
        chirp_end = chirp_bounds[2];
        chirp_start = chirp_bounds[1];
        # plot a vertical line
        plot!(audioindextoms.([chirp_start, chirp_start]), [1, size(Sx, 1)], linewidth=5, color=end_color, label="Chirp bounds")
        plot!(audioindextoms.([chirp_end, chirp_end]), [1, size(Sx, 1)], linewidth=5, color=end_color, label=false)
    end
    return p;
end

"""
    plotmelodydb(chirp_seq_single_mic::ChirpSequence, melody::Vector{Int64};
        nfft=256, bandpass_filter=(20_000, 100_000))

Plots the strength of the melody estimated by `findmelody`, in decibels.

Inputs:
- `chirp_seq_single_mic`: input chirp sequence object (for a single microphone).
- `melody`: result of `findmelody`.
- `nfft` (default: 256): window size for computing the spectrogram.
- `bandpass_filter` (default: `(20_000, 100_000)`): before computing the melody,
    a band-pass filter is applied to the spectrogram. `bandpass_filter` is a
    tuple of the (lower cutoff, upper cutoff), in Hertz.
"""
function plotmelodydb(chirp_seq_single_mic::ChirpSequence, melody::Vector{Int64};
        nfft=256, bandpass_filter=(20_000, 100_000))
    pad_len = Int64(round(nfft/2));
    x = vcat(zeros(pad_len), chirp_seq_single_mic.mic_data, zeros(nfft-1-pad_len));
    Sx = stft(x, nfft, nfft-1, window=hamming(nfft));
    Sx = bandpassfilterspecgram(Sx, bandpass_filter[1], bandpass_filter[2], nfft=nfft);
    Sx_db = pow2db.(abs.(Sx) .^ 2); # in decibels

    N = length(melody);
    melody_db = [Sx_db[melody[i], i] for i=1:N];
    return plotmicdata(melody_db; title="Power (db) of the melody", ylabel="Power")
end

"""
    separatechirpkwargs(;chirp_kwargs...) -> Dict{Symbol, Any},  
                                             Dict{Symbol, Any},
                                             Dict{Symbol, Any}

Given keyword arguments from a combination of `findmelody`,
`estimatechirpbounds`, and `computemelodyoffsets`, separate them into  three
dictionaries: one for each of those functions.
"""
function separatechirpkwargs(;chirp_kwargs...)
    melody_kwargs = Dict{Symbol, Any}();
    chirp_bound_kwargs = Dict{Symbol, Any}();

    common_keys = [:nfft, :bandpass_filter];
    melody_keys = [common_keys..., :maximum_melody_slope, :melody_drop_thresh_db];
    chirp_bound_keys =[common_keys..., :melody_drop_thresh_db, :melody_thresh_db_low, :moving_avg_size];
    offset_keys = [:max_offset, :tolerance];

    for key=melody_keys
        if haskey(chirp_kwargs, key)
            melody_kwargs[key] = chirp_kwargs[key];
        end
    end
    for key=chirp_bound_keys
        if haskey(chirp_kwargs, key)
            chirp_bound_kwargs[key] = chirp_kwargs[key];
        end
    end
    
    offset_kwargs = merge(melody_kwargs, chirp_bound_kwargs);
    for key=offset_keys
        if haskey(chirp_kwargs, key)
            offset_kwargs[key] = chirp_kwargs[key];
        end
    end
    return melody_kwargs, chirp_bound_kwargs, offset_kwargs;
end

"""
    estimatechirp(chirp_seq_single_mic::ChirpSequence, peak_snr_thresh::Real;
    chirp_kwargs...) -> Int64, Vector{Float64}

Use `estimatechirpend` to separate the chirp from the echos (for a single
`ChirpSequence` object) by returning the chirp sequence up until the estimated
end index.

Inputs:
- `chirp_seq_single_mic`: input chirp sequence object (for a single microphone).
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `chirp_kwargs`: you can additionally pass in any keyword arguments for
    `findmelody` and/or `estimate_chirp_bounds`.

Output:
- `chirp_start`: index of the chirp start, in samples since the beginning of
    the chirp sequence.
- `chirp_est`: audio data of the estimated chirp.
"""
function estimatechirp(chirp_seq_single_mic::ChirpSequence, peak_snr_thresh::Real;
    chirp_kwargs...)
    
    melody_kwargs, chirp_bound_kwargs, offset_kwargs = separatechirpkwargs(;chirp_kwargs...);

    melody = findmelody(chirp_seq_single_mic, peak_snr_thresh; melody_kwargs...);
    chirp_start, chirp_end = estimatechirpbounds(chirp_seq_single_mic, melody,
        peak_snr_thresh; chirp_bound_kwargs...);

    return chirp_start, chirp_seq_single_mic.mic_data[chirp_start:chirp_end];
end

"""
    plotestimatedchirps(chirp_seq_all_mics::Dict{Int64, ChirpSequence},
        peak_snr_thresh::Real; nfft=256, bandpass_filter=(20_000, 100_000),
        maximum_melody_slope=5, melody_drop_thresh_db=20,
        melody_thresh_db_low=-20, moving_avg_size=10)

Plots the result of `estimatechirp`, for every microphone present in
`chirp_seq_all_mics`, by plotting the estimated chirp on top of the chirp
sequence.

Inputs:
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- Rest of the arguments: see `estimatechirp`.
"""
function plotestimatedchirps(chirp_seq_all_mics::Dict{Int64, ChirpSequence}, peak_snr_thresh::Real;
    chirp_kwargs...)

    num_plots = Int64(length(chirp_seq_all_mics) + length(chirp_seq_all_mics) % 2);
    plots = Matrix(undef, num_plots, 1);

    chirp_lens = zeros(length(chirp_seq_all_mics));

    for (i, mic)=enumerate(sort(collect(keys(chirp_seq_all_mics))))
        seq_i = chirp_seq_all_mics[mic];
        start_idx, estimated_chirp = estimatechirp(seq_i, peak_snr_thresh; chirp_kwargs...);

        plot_idxs = seq_i.start_idx:(seq_i.start_idx + seq_i.length - 1);

        plots[i] = plotmicdata(seq_i.mic_data, plot_idxs=plot_idxs, 
            title=(@sprintf "Chirp Sequence and chirp for mic %d" mic); 
            label="Sequence");

        chirp_L = length(estimated_chirp);
        chirp_lens[i] = audioindextoms(chirp_L);
        plot!(audioindextoms.(plot_idxs[start_idx:start_idx+chirp_L-1]), estimated_chirp; label="Estimated chirp");

    end

    for i = length(chirp_seq_all_mics)+1:num_plots
        plots[i] = myplot([0, 0], legend=false, title="Blank Plot", xlabel="", ylabel="");
    end
    println("Chirp Lengths (ms): ", chirp_lens)
    return plot(plots..., layout=(Int64(num_plots / 2), 2), size=(1100, 150*num_plots))
end

"""
    computemelodyoffsets(chirp_sequence_all_mics::Dict{Int64, ChirpSequence},
        peak_snr_thresh::Real; max_offset=500, max_negative_offset=100,
        tolerance=1, chirp_kwargs...)

Sometimes, especially for noisy data, the beginnings of the chirps get cut off.
This function takes in data from all (high-snr) microphones for a chirp
sequence and estimates how many samples were cut off from the beginning of
each chirp.

It does this by:
1. Finding the mic with the highest SNR. This mic will be used as reference.
2. Finding the melody and chirp lengths for each microphone.
3. Shifting the chirps for the non-reference microphones forward and backward
    in time until we find the shift that minimizes the distance between that
    chirp's melody and the reference microphone's melody.

This ensures that we can "align" all of the chirps.

Inputs:
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `max_offset` (default: 500): maximum shift forward to try, in audio samples
    (i.e., if the beginning of the chirp was cut off).
- `max_negative_offset` (default: 100): maximum shift backward to try, in audio
    samples (i.e., if there is some noise picked up before the beginning of the
    actual chirp). This should be relatively small to avoid erroneously cutting
    off the beginnings of chirps.
- `tolerance` (default: 1): if the offset found in step 3 does not decrease the
    mean absolute error of the difference between the reference melody and the
    given microphone's melody by at least `tolerance`, then just set the offset
    to zero.
- `chirp_kwargs`: you can additionally pass in any keyword arguments for
    `findmelody` and/or `estimate_chirp_bounds`.

Output:
- `offsets`: dictionary mapping microphone number to the best shift, in samples
    of audio data, found in part 3.
"""
function computemelodyoffsets(chirp_sequence_all_mics::Dict{Int64, ChirpSequence},
    peak_thresh::Real; max_offset=500, max_negative_offset=100, tolerance=2, chirp_kwargs...)

    melody_kwargs, chirp_bound_kwargs, _ = separatechirpkwargs(;chirp_kwargs...);

    ## test refining the chirp sequences where the beginnings are cut off
    melodies = Dict{Int64, Vector{Int64}}();
    chirp_ends = Dict{Int64, Int64}();
    offsets = Dict{Int64, Int64}();

    highest_snr = -Inf;
    reference_mic = 0;
    for mic_idx = keys(chirp_sequence_all_mics)
        snr = maximum(chirp_sequence_all_mics[mic_idx].snr_data);
        if snr > highest_snr
            reference_mic = mic_idx;
            highest_snr = snr;
        end
        melodies[mic_idx] = findmelody(chirp_sequence_all_mics[mic_idx], peak_thresh;  melody_kwargs...);
        chirp_start, chirp_ends[mic_idx] = estimatechirpbounds(chirp_sequence_all_mics[mic_idx],
                melodies[mic_idx], peak_thresh; chirp_bound_kwargs...);
        melodies[mic_idx] = melodies[mic_idx][chirp_start:end];
        chirp_ends[mic_idx] -= (chirp_start - 1);
    end

    for mic_idx = keys(chirp_sequence_all_mics)
        chirp_end = min(chirp_ends[reference_mic], chirp_ends[mic_idx])
        offset_options = -min(max_negative_offset, chirp_end):max(min(max_offset, chirp_ends[reference_mic] - chirp_ends[mic_idx]-1), 0)
        
        if mic_idx == reference_mic || length(offset_options) == 0
            offsets[mic_idx] = 0;
            continue;
        end
            
        offsets[mic_idx] = argmin([mean(abs.(
            melodies[reference_mic][max(1, 1+i):i+chirp_end] - 
            melodies[mic_idx][1:chirp_end + i - max(0, i)]))
                for i=offset_options]) + offset_options[1];
        i = offsets[mic_idx];
        error_offset = mean(abs.(
            melodies[reference_mic][max(1, 1+i):i+chirp_end] -
            melodies[mic_idx][1:chirp_end + i - max(0, i)]));
        error_zero = mean(abs.(melodies[reference_mic][1:chirp_end] - melodies[mic_idx][1:chirp_end]));

        if error_zero - error_offset <= tolerance
            offsets[mic_idx] = 0;
        end
    end

    
    return offsets;
end

"""
function plotoffsetchirps(chirp_seq_all_mics::Dict{Int64, ChirpSequence}, 
    offsets::Dict{Int64, Int64})

Plot a chirp sequence, where the data from each microphone is shifted by the
value found in `computemelodyoffsets`. Plots spectrograms in a vertical layout
so that you can visially see if the chirps are aligned with each other.

Inputs:
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- `offsets`: output of `computemelodyoffsets`.
- `start_idxs`: dictionary mapping microphone indices to their respective
    chirp start indices (in samples since the start of the chirp sequence).
    You can get this using `getchirpstartandendindices`.
"""
function plotoffsetchirps(chirp_seq_all_mics::Dict{Int64, ChirpSequence}, offsets::Dict{Int64, Int64}, 
        start_idxs::Dict{Int64, Int64})
    n_mics = length(chirp_seq_all_mics);
    sorted_mics = sort(collect(keys(chirp_seq_all_mics)));

    num_plots = Int64(n_mics);
    plots = Matrix(undef, num_plots, 1);

    longest_seq = 0;
    for mic=sorted_mics
        longest_seq = max(longest_seq, chirp_seq_all_mics[mic].length + offsets[mic]);
    end

    for (i, mic)=enumerate(sorted_mics)
        seq_i = chirp_seq_all_mics[mic];

        plot_data = nothing;
        if offsets[mic] >= 0
            plot_data = vcat(zeros(offsets[mic]), seq_i.mic_data[start_idxs[mic]:end], zeros(longest_seq - offsets[mic] - seq_i.length));
        else
            plot_data = vcat(seq_i.mic_data[start_idxs[mic]-offsets[mic]:end], zeros(longest_seq - offsets[mic] - seq_i.length));
        end
        plots[i] = plotSTFTtime(plot_data, nfft=256, noverlap=255, title=(@sprintf "Isolated chirp sequence for mic %d" mic));
    end
    return plot(plots..., layout=(num_plots, 1), size=(1100, 300*num_plots))
end