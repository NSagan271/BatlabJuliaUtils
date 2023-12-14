using DataInterpolations;
using Roots;
using Printf;

"""
    struct ChirpSequence
        start_idx::Int64
        length::Int64
        vocalization_time_ms::Float64
        snr_data::Vector{Float64}
        mic_data::Vector{Float64}
        mic_num::Int64
    end

Datastructure to store individual chirp sequences (for a single microphone).

Fields:
- `start_idx`: start of the chirp, in numner of samples since the beginning of
    the audio data.
- `length`: length, in audio samples, of the chirp sequence.
- `vocalization_time_ms`: time (since the beginning of the audio data) that the
    bat made the vocalization. Estimated using the centroid data.
- `snr_data`: vector of estimated SNR values over the duration of the chirp
    sequence. Produced by `estimatesnr`.
- `mic_data`: audio data for the given microphone, over the duration of the
    chirp sequence.
- `mic_num`: which microphone (from 1 to 4) the data corresponds to.
"""
struct ChirpSequence
    start_idx::Int64
    length::Int64
    vocalization_time_ms::Float64
    snr_data::Vector{Float64}
    mic_data::Vector{Float64}
    mic_num::Int64
end

"""
    getboundsfromboxes(boxes; filter_fn=(start_idx, stop_idx) -> true)
        -> Matrix{Int64}

Given bitarray `boxes` (as a column vector), return a matrix where the first column
is the start indices of the sections where `boxes==1`, and the second column is the
end indices of those sections.

Optionally, only keep boxes where filter_fn returns true.

Inputs:
- `boxes`: one-dimensional bitarray.
- `filter_fn` (default: always return `true`): function that takes in the start
    and end indices of a region where `boxes==1` and returns `false` for boxes
    to discard.

Output:
- `bounds`: two-column matrix, as described above.
"""
function getboundsfromboxes(boxes; filter_fn=(start_idx, stop_idx) -> true) :: Matrix{Int64}
    # If boxes is one-dimensional, make it a matrix with a single column.
    # Otherwise, it'll be unchanged.
    boxes = reshape(boxes, (size(boxes, 1), :));
    bounds = Matrix{Int64}(undef, 0, 2)
    i = 1;
    while i <= length(boxes)
        start_idx = findfirst(boxes[i:end]);
        if isnothing(start_idx)
            break;
        end
        start_idx = start_idx[1] + i - 1;
        stop_idx = findfirst(boxes[start_idx:end] .== 0);
        stop_idx = isnothing(stop_idx) ? length(boxes) : start_idx + stop_idx[1] - 1;
    
        if filter_fn(start_idx, stop_idx)
            bounds = vcat(bounds, [start_idx stop_idx]);
        end
        i = stop_idx + 1;
    end
    return bounds;
end

"""
    findhighsnrregions(snr::AbstractArray; signal_thresh::Float64,
        peak_thresh::Float64, maxfilter_length::Int64) -> BitArray

Given the estimated SNR of the audio data (from the function `estimatesnr`),
determine which regions are likely to contain chirp sequences, as follows:
1. Apply a `maxfilter` to the `snr` array: this helps us find contiguous
    regions with high SNR.
2. Find all sections where the maxfiltered SNR is above `signal_thresh`.
3. Of those sections, keep the ones where, at some point, the SNr goes above
    `peak_thresh`.

Inputs:
- `snr`: estimated SNR of the audio data, produced by `estimatesnr`, either the
    full matrix or one column.
- `signal_thresh`, `peak_thresh`: thresholds, described above.
- `maxfilter_length`: half-length, in samples, of the maximum filter.

Output:
- `high_snr_locations`: bitarray that is `1` in regions that likely contain
    chirp sequences.
"""
function findhighsnrregions(snr::AbstractArray, signal_thresh::Number,
        peak_thresh::Number, maxfilter_length::Int64) :: BitArray
    
    # If snr is one-dimensional, make it a matrix with a single column.
    # Otherwise, it'll be unchanged.
    snr = reshape(snr, (size(snr, 1), :));
    maxfiltered_snr = maxfilter(snr, maxfilter_length);
    signal_locations = maxfiltered_snr .> signal_thresh;

    high_snr_locations = BitArray(undef, size(snr));
    high_snr_locations .= 0;
    for mic=1:size(snr, 2)
        # Find all regions that 
        high_snr_bounds = getboundsfromboxes(signal_locations[:, mic]; 
            filter_fn=(start_idx, stop_idx) -> maximum(maxfiltered_snr[start_idx:stop_idx, mic]) > peak_thresh);
        for row=1:size(high_snr_bounds, 1)
            high_snr_locations[high_snr_bounds[row, 1]:high_snr_bounds[row, 2], mic] .= 1;
        end
    end
    return high_snr_locations;
end

"""
    findroughchirpsequencebounds(snr::Matrix{Float64}, mic::Int64,
        signal_thresh::Number, peak_thresh::Number, maxfilter_length::Int64)
                                                            -> Matrix{Int64}

For a single microphone/channel, converts the output of `findhighsnrregions` to
a two-column matrix, where the first column is the start index of each
presumed chirp sequence.

Inputs:
- `snr`: estimated SNR of the audio data, produced by `estimatesnr` (full 
    matrix).
- `signal_thresh`, `peak_thresh`: thresholds, described in
    `findhighsnrregions`.
- `maxfilter_length`: half-length, in samples, of the maximum filter.

Output: described above.
"""
function findroughchirpsequenceidxs(snr::Matrix{Float64}, mic::Int64, signal_thresh::Number,
        peak_thresh::Number, maxfilter_length::Int64) :: Matrix{Int64}
    high_snr_locations = findhighsnrregions(snr[:, mic], signal_thresh, peak_thresh, maxfilter_length);
    result = getboundsfromboxes(high_snr_locations);

    ## Adjust for the max-filtering done.
    result[:, 1] = result[:, 1] .+ maxfilter_length .- 1;
    result[:, 2] = result[:, 2] .- maxfilter_length .+ 1;
    return result;
end

"""
    adjustsequencebounds(snr::Matrix{Float64}, mic::Int64,
        rough_bounds::Vector{Int64}, max_end_idx::Int64; tail_snr_thresh::Real,
        max_seq_len=MAX_SEQUENCE_LENGTH, maxfilter_seq_end=50) -> Vector{Int64}
    
Given rough bounds for a single chirp sequence (one row of the output of
`findroughchirpsequenceidxs`), adjust the end index to ensure that the chirp
sequence isn't cut off early.

The process is similar to `findhighsnrregions`, but with more lenient
thresholds.
1. Apply a `maxfilter` to the `snr` array, with a filter size that is ideally
    longer than the one used for `findroughchirpsequenceidxs`.
2. Find the first index of the maxfiltered SNRs that goes below
    `tail_snr_thresh`, a threshold lower than the one used for
    `findroughchirpsequenceidxs`, and sets the end index of the chirp
    sequence to this. If this index is beyond `max_end_idx`, then the end of
    the chirp sequence is set to `max_end_idx`.

Inputs:
- `snr`: estimated SNR of the audio data, produced by `estimatesnr` (full 
    matrix).
- `mic`: microphone number, from 1 to 4.
- `rough_bounds`: row of the matrix produced by `findroughchirpsequenceidxs`.
- `max_end_idx`: the start of the next chirp sequence, or the end of the signal
        if this is the last chirp sequence for a particular microphone.
- `tail_snr_thresh`: described above.
- `max_seq_len` maximum length of the chirp sequence. Default set in
    `Defaults.jl`.
- `maxfilter_length` (default: 50): half-length, in samples, of the maximum
    filter.

Output:
- `refined_bounds`: two-element vector, where the first element is the start
    of the chirp sequence (unchanged), and the second element is the 
"""
function adjustsequenceidxs(snr::Matrix{Float64}, mic::Int64, rough_bounds::Vector{Int64}, max_end_idx::Int64,
        tail_snr_thresh::Real; max_seq_len=MAX_SEQUENCE_LENGTH, maxfilter_length=50) :: Vector{Int64}
    refined_bounds = copy(rough_bounds);

    max_end_idx = min(max_end_idx, refined_bounds[1] + max_seq_len);
    
    seq_end = findfirst(maxfilter(snr[rough_bounds[2]:max_end_idx, mic], maxfilter_length) .<= tail_snr_thresh);
    seq_end = isnothing(seq_end) ? max_end_idx : rough_bounds[2] + seq_end[1] - 1;
    refined_bounds[2] = min(seq_end, max_end_idx);

    return refined_bounds;
end

"""
    getvocalizationtimems(chirp_start_index::Int64, mic::Int64,
        location_data::Matrix{Float64}, mic_positions::Matrix{Float64}; 
        buffer_time_ms=100, fs_video=360,
        interp_type=QuadraticInterpolation) -> Float64

Given the index of the audio data at which a chirp sequence starts, estimate
the time that the bat made a vocalization, in milliseconds since the start of
the audio data.

This is achieved by taking a slice of the centroid data of radius `buffer_time_ms`
around the time the chirp reached the microphone, performing quadratic
interpolation of the centroid data over that slice, and solving for `t` in

`distance_from_mic(t) = speed_of_sound * (time_chirp_reached_mic - t)`

to get the vocalization time.

This function returns `NaN` (not a number) if there isn't sufficient location
data to determine the vocalization time.

Inputs:
- `chirp_start_index`: index of the audio data that the chirp sequence started.
- `mic`: microphone that heard the chirp sequence.
- `location_data`: full centroid data, where the columns represent coordinates
    (x, y, z).
- `mic_positions`: matrix of microphone positions, where each row is a
    different microphone and the columns represent coordinates (x, y, z).
- `buffer_time_ms` (default: 100): radius, in milliseconds, of the slice of
    centroid data to examine.
- `fs_video` (default set in `Defaults.jl`): sampling rate of the centroid
    data.
- `interp_type` (default: `QuadraticInterpolation`): type of interpolation
    (from the package `DataInterpolations`).
"""
function getvocalizationtimems(chirp_start_index::Int64, mic::Int64, location_data::Matrix{Float64}, mic_positions::Matrix{Float64}; 
        buffer_time_ms=100, fs_video=FS_VIDEO, interp_type=QuadraticInterpolation) :: Float64

    chirp_reached_mic_sec = audioindextosec(chirp_start_index); # in seconds since the start of the audio data
    if chirp_reached_mic_sec >= 8 # the chirp reached the microphone after the end of
                            # the audio data, so we give up
        return NaN;
    end

    # Take a slice of video data around chirp_reached_mic of radius buffer_time_ms
    slice_start_time = max(chirp_reached_mic_sec - buffer_time_ms / 1000, 0);
    slice_end_time = min(chirp_reached_mic_sec + buffer_time_ms / 1000, 8);

    slice_start_idx = sectovideoindex(slice_start_time, size(location_data, 1));
    if slice_start_idx < 1 # the slice starts before the beginning of the video data,
                            # so we give up
        return NaN;
    end

    slice_idxs, video_data_slice = getvideoslicefromtimes(location_data, slice_start_time, slice_end_time);
    if any(isnan.(video_data_slice)) # the video data has NaNs, so we give up
        return NaN;
    end

    L = size(location_data, 1);

    # Perform interpolation on the video data
    interp = interp_type(vec(distancefrommic(video_data_slice, mic_positions, mic)) ./ 1000, videoindextosec.(slice_idxs, L));

    # The vocalization time is the solution to:
    #   distance_from_mic(t) = speed_of_sound * (time_chirp_reached_mic - t),
    # or the root (zero crossing) of
    #    distance_from_mic(t) + speed_of_sound * (t - time_chirp_reached_mic).
    f = t -> interp(t) .+ SPEED_OF_SOUND .* (t - chirp_reached_mic_sec);

    return find_zero(f, chirp_reached_mic_sec) * 1000; # convert to milliseconds
end

"""
    groupchirpsequencesbystarttime(chirp_sequence_bounds_per_mic::Matrix{Int64},
        snr::Matrix{Float64}, y::Matrix{Float64},
        location_data::Matrix{Float64}, mic_locations::Matrix{Float64},
        single_chirp_snr_thresh=100, vocalization_start_tolerance_ms=1.5) 
                    -> Vector{Dict{Int64, ChirpSequence}}, Vector{Float64}

Given start and end indices of chirp sequences, for all microphones,
determine which chirp sequences came from the same initial chirp. Only keep
chirp sequences arising from vocalizations heard by at least two microphones.

Inputs:
- `chirp_sequence_bounds_per_mic`: array of `[chirp sequence indices for mic 1,
    ..., chirp sequence indices for mic 4]`."Chirp sequence indices for
    mic i" is a two-column matrix where the first column is the start indices
    of each chirp sequence and the second column is the corresponding end
    indices.
- `snr`: estimated SNR of the audio data, produced by `estimatesnr`.
- `y`: matrix of audio data, where each column is a different microphone.
- `location_data`: full centroid data, where the columns represent coordinates
    (x, y, z).
- `mic_positions`: matrix of microphone positions, where each row is a
    different microphone and the columns represent coordinates (x, y, z).
- `single_mic_snr_thresh` (default: 100): if a chirp sequence only has data
    from one microphone, still store the chirp sequence if it has a SNR over
    this value.
- `vocalization_start_tolerance_ms` (default: 1.5): if the estimated
    vocalization time for two chirp sequences (for different microphones) is
    within `vocalization_start_tolerance_ms` milliseconds, then they are
    considered to be from the same vocalization.

Output:
- `chirp_sequences`: example form
    ```
    [{1 -> ChirpSequences(...), 2 -> ChirpSequence(...)},
     {2-> ChirpSequence(...), 4 -> ChirpSequence(...), 3 -> ChirpSequence(...)},
     ...
    ]
    ```
    Each element of the `chirp_sequences` vector corresponds to all chirp
    sequences arising from a single vocalization. This is represented by a
    dictionary mapping microphone number to `ChirpSequence` structure.

- `vocalization_times`: vector, where each element is the estimated vocalization
    time for the corresponding chirp sequence.
"""
function groupchirpsequencesbystarttime(chirp_sequence_bounds_per_mic::Array{Matrix{Int64}}, snr::Matrix{Float64},
        y::Matrix{Float64}, location_data::Matrix{Float64}, mic_locations::Matrix{Float64}; single_mic_snr_thresh=100,
        vocalization_start_tolerance_ms=1.5)
    ### Data structures to return:
    #################################################################
    # chirp_sequences: example
    # [{1 -> ChirpSequences(...), 2 -> ChirpSequence(...)},
    #  {2-> ChirpSequence(...), 4 -> ChirpSequence(...), 3 -> ChirpSequence(...)},
    #  ...]
    # Described in the docstring.
    chirp_sequences = Vector{Dict{Int64, ChirpSequence}}(undef, 0);

    # vocalization_times: vector, where each element is the estimated vocalization
    # time of the corresponding chirp sequence.
    vocalization_times = Vector{Float64}(undef, 0);
    #################################################################

    # Algorithm summary:
    # current_seq_idx = [1, 1, 1, 1] # chirp sequence we're looking at, for
    #                                # for each microphone. Start at the beginning.
    # current_seq = { empty dictionary } # will be a mapping from 
    #                                    # mic number -> ChirpSequence
    # current_vocalization_time = 0
    #
    # loop until we go through all chirp sequences:
    #   chirp_times = for each mic, the estimated vocalization time of the
    #                 chirp sequence we're currently examining
    #   mic = microphone with the earliest vocalization time, for the set of
    #         chirp sequences we're examining at this step
    #
    #   if the corresponding vocalization time is within 
    #   vocalization_start_tolerance_ms of current_vocalization_time:
    #       1. Add the current chirp sequence heard by mic to current_seq
    #       2. Update current_vocalization_time to be the average of the estimated
    #           vocalization times over current_seq.
    #   else:
    #       1. If current_seq has data from at least two microphones, add it to
    #           chirp_sequences (which will be eventually returned). Add
    #           current_vocalization_time to vocalization_times.
    #       2. Reset current_seq and add the current chirp sequence heard by mic.
    #       3. Set current_vocalization_time to the vocalization time of the
    #           chirp sequence.
    #   Increment current_seq_idx for mic (in the next iteration, look at the
    #       next chirp sequence for this microphone).

    n_mics = size(y, 2);
    current_seq_idx = ones(Int64, n_mics);

    # number of chirp sequences detected for each microphone
    num_seqs = [size(chirp_sequence_bounds_per_mic[mic], 1) for mic=1:n_mics];
    current_vocalization_time = 0;
    current_seqs = Dict{Int64, ChirpSequence}();

    while any(current_seq_idx .<= num_seqs) # while there are still chirp sequences left
        # estmate the vocalization times
        no_nans = false;
        chirp_times = nothing;
        while !no_nans
            chirp_times = [
                (current_seq_idx[mic] <= num_seqs[mic]) ?
                getvocalizationtimems(chirp_sequence_bounds_per_mic[mic][current_seq_idx[mic], 1], 
                                        mic, location_data, mic_locations) : Inf
                for mic=1:n_mics
            ];

            no_nans = true;
            for mic=1:n_mics
                # If any vocalization times couldn't be estimated, skip that
                # chirp sequence
                if isnan(chirp_times[mic])
                    no_nans = false;
                    current_seq_idx[mic] += 1;
                end
            end
        end

        # Find the first chirp sequence, among the ones we're currently looking at
        mic = argmin(chirp_times);
        start_time = chirp_times[mic];

        if isinf(start_time)
            break;
        end

        # The current chirp sequence comes from a different vocalization than 
        if abs(start_time - current_vocalization_time) >= vocalization_start_tolerance_ms
            if length(current_seqs) >= 2 || ((length(current_seqs) == 1) && 
                    maximum(first(values(current_seqs)).snr_data) >= single_mic_snr_thresh)
                push!(vocalization_times, current_vocalization_time);
                push!(chirp_sequences, current_seqs);
            end  
            # clear current_seqs       
            current_seqs = Dict{Int64, ChirpSequence}();
        end
        
        if !haskey(current_seqs, mic)
            # average current vocalization time among all sequences in 
            # [current_seqs + new chirp sequence to add)
            current_vocalization_time = (current_vocalization_time * length(current_seqs) + start_time) /
                                            (length(current_seqs) + 1);
            # Make a new ChirpSequence structure
            start_idx = chirp_sequence_bounds_per_mic[mic][current_seq_idx[mic], 1];
            stop_idx = chirp_sequence_bounds_per_mic[mic][current_seq_idx[mic], 2];
            current_seqs[mic] = ChirpSequence(
                start_idx, stop_idx-start_idx+1, start_time, snr[start_idx:stop_idx, mic],
                y[start_idx:stop_idx, mic], mic);
        end
        current_seq_idx[mic] += 1;
    end
    return chirp_sequences, vocalization_times;
end

"""
    function plotchirpsequence(chirp_seq::Dict{Int64, ChirpSequence},
        location_data::Matrix{Float64}; plot_separate=false,
        plot_spectrogram=false)
    
Plots a chirp sequence on one of three formats:
1. If `plot_separate` and `plot_spectrogram` are both `false`, it plots the
    data from all microphones in the same plot.
2. If `plot_spectrogram` is `true`, then it plots the spectrogram of the
    chirp sequence for each microphone (in separate plots).
3. Otherwise, if `plot_separate` is `true`, it plots the time-domain
    waveforms for each microphone (in separate plots).
    
Inputs:
- `chirp_seq_all_mics`: dictionary mapping microphone number to a
    `ChirpSequence` object.
- `plot_separate`, `plot_spectrogram`: descibed above.
"""
function plotchirpsequence(chirp_seq_all_mics::Dict{Int64, ChirpSequence}; plot_separate=false, plot_spectrogram=false)
    ## For plotting separately
    n_mics = length(chirp_seq_all_mics);
    sorted_mics = sort(collect(keys(chirp_seq_all_mics)));

    retval = nothing;

    num_plots = Int64(n_mics + n_mics % 2);
    plots = Matrix(undef, num_plots, 1);

    for (i, mic)=enumerate(sorted_mics)
        seq_i = chirp_seq_all_mics[mic];
        plot_idxs = seq_i.start_idx:seq_i.start_idx+seq_i.length-1;

        if plot_spectrogram
            plots[i] = plotSTFTtime(seq_i.mic_data, nfft=256, noverlap=255, title=(@sprintf "Isolated chirp sequence for mic %d" mic));
        elseif plot_separate
            plots[i] = plotmicdata(seq_i.mic_data,  plot_idxs=plot_idxs , title=(@sprintf "Isolated chirp sequence for mic %d" mic));
        elseif i == 1
            retval = plotmicdata(seq_i.mic_data; plot_idxs=plot_idxs, label=(@sprintf "mic %d" mic), title="Chirp Sequence");
        else
            plot!(audioindextoms.(plot_idxs), seq_i.mic_data, label=(@sprintf "mic %d" mic))
        end
    end

    if plot_separate || plot_spectrogram
        for i = n_mics+1:num_plots
            plots[i] = myplot([0, 0], legend=false, title="Blank Plot", xlabel="", ylabel="");
        end
        return plot(plots..., layout=(Int64(num_plots / 2), 2), size=(1100, 150*num_plots))
    end
    return retval;
end