using DataInterpolations;
using Roots;
using Printf;

"""
    function estimatebuzzphase(snr::AbstractArray;
        buzz_phase_chirp_separation_ms=20) -> Matrix{Int}

Makes a very rudimentary estimate of chirp onsets, as heard by microphones, and
then finds all times where chirp onsets are less than
`buzz_phase_chirp_separation_ms` apart, for each microphone. It then prints out
these time windows in descending order of length (these are the potential buzz
phase times).

Note that can also classify some echos as chirp sequences, so expect there to
be false positives.

Inputs:
- `snr`: output of `estimatesnr`.
- `buzz_phase_chirp_separation_ms` (default: 20): maximum number of
    milliseconds expected between onsets of consecutive buzz phase chirps.
"""
function estimatebuzzphase(snr::AbstractArray; buzz_phase_chirp_separation_ms=20, fs=FS) :: Matrix{Int}
    possible_buzz_phase = zeros(size(snr, 1));
    for mic=1:4
        bounds = findhighsnrregionidxs(snr, 1, 30, 40, 30)
        for i=2:size(bounds, 1)
            if audioindextoms(bounds[i, 1] - bounds[i-1, 1]) <= buzz_phase_chirp_separation_ms
                possible_buzz_phase[bounds[i-1, 1]:bounds[i, 2]] .= 1;
            end
        end
    end

    for min_length_ms=50:-10:0
        possible_buzz_phase_idxs = getboundsfromboxes(possible_buzz_phase .== 1;
            filter_fn=(start_idx, stop_idx) -> (stop_idx - start_idx + 1) >= min_length_ms*FS/1000);
        if length(possible_buzz_phase_idxs) > 0
            break;
        end
    end
    possible_buzz_phase_idxs = possible_buzz_phase_idxs[sortperm(possible_buzz_phase_idxs[:, 1] - possible_buzz_phase_idxs[:, 2]), :];

    if length(possible_buzz_phase_idxs) == 0
        println("Could not find any buzz phase here!");
        return;
    end

    println("Printing possible buzz phase indices, in order of most to least probable\n(aka, longest to shortest)\n------------------------");
    for i=1:size(possible_buzz_phase_idxs, 1)
        onset = possible_buzz_phase_idxs[i, 1];
        offset = possible_buzz_phase_idxs[i, 2];
        len = audioindextoms(offset - onset + 1);
        (@printf "From %d to %d (%d to %d millis): %d millis in length\n" onset offset round(audioindextoms(onset)) round(audioindextoms(offset)) round(len));
    end
    println("------------------------");
    return possible_buzz_phase_idxs;
end

"""
    struct ChirpSequence
        start_idx::Int
        length::Int
        vocalization_time_ms::Real
        snr_data::Vector{Real}
        mic_data::Vector{Real}
        mic_num::Int
    end

Datastructure to store individual chirp sequences (for a single microphone).

Fields:
- `start_idx`: start of the chirp, in number of samples since the beginning of
    the audio data that this chirp sequence comes from (i.e., the MAT data that
    was initially read in).
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
    start_idx::Int
    length::Int
    vocalization_time_ms::Real
    snr_data::Vector{Real}
    mic_data::Vector{Real}
    mic_num::Int
end

"""
    getboundsfromboxes(boxes::AbstractArray;
        filter_fn=(start_idx, stop_idx) -> true) -> Matrix{Int}

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
function getboundsfromboxes(boxes::AbstractArray; filter_fn=(start_idx, stop_idx) -> true) :: Matrix{Int}
    # If boxes is one-dimensional, make it a matrix with a single column.
    # Otherwise, it'll be unchanged.
    boxes = vectortomatrix(boxes);
    bounds = Matrix{Int}(undef, 0, 2);
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
    findhighsnrregions(snr::AbstractArray, signal_thresh::Number,
        min_peak_thresh::Number, maxfilter_length::Int;
        snr_drop_thresh=20, peak_snr_thresh_radius=1500)-> BitArray

Given the estimated SNR of the audio data (from the function `estimatesnr`),
determine which regions are likely to contain chirp sequences, as follows:
1. Apply a `maxfilter` to the `snr` array: this helps us find contiguous
    regions with high SNR.
2. Find all sections where the maxfiltered SNR is above `signal_thresh`.
3. Of those sections, keep the ones where, at some point, the SNR goes above
    another, higher threshold.

The threshold in step 3 above is set as follows:
1. We look around in a range of `peak_snr_thresh_radius` around the "signal
    region" and we find the maximum SNR present in that range.
2. We require that the peak SNR of the "signal region" be at most
    `snr_drop_thresh` dB lower than that maximum value.
3. We also require that the peak SNR be no lower than `min_peak_thresh`.

Inputs:
- `snr`: estimated SNR of the audio data, produced by `estimatesnr`, either the
    full matrix or one column.
- `signal_thresh`, `min_peak_thresh`, `snr_drop_thresh`,
    `peak_snr_thresh_radius`: thresholds, descsribed above.
- `maxfilter_length`: half-length, in samples, of the maximum filter.

Output:
- `high_snr_locations`: bitarray that is `1` in regions that likely contain
    chirp sequences.
"""
function findhighsnrregions(snr::AbstractArray, signal_thresh::Number,
        min_peak_thresh::Number, maxfilter_length::Int;
        snr_drop_thresh=20, peak_snr_thresh_radius=1500) :: BitArray

    # If snr is one-dimensional, make it a matrix with a single column.
    # Otherwise, it'll be unchanged.
    snr = vectortomatrix(snr);
    maxfiltered_snr = maxfilter(snr, maxfilter_length);
    signal_locations = maxfiltered_snr .> signal_thresh;

    high_snr_locations = BitArray(undef, size(snr));
    high_snr_locations .= 0;
    for mic=1:size(snr, 2)
        # Find all regions that 
        new_peak_thresh = (start_idx, stop_idx) -> max(min_peak_thresh, 
            maximum(maxfiltered_snr[max(start_idx-peak_snr_thresh_radius, 1):min(stop_idx+peak_snr_thresh_radius, size(snr, 1)), mic]) -
                    snr_drop_thresh);
        high_snr_bounds = getboundsfromboxes(signal_locations[:, mic]; 
            filter_fn=(start_idx, stop_idx) -> maximum(maxfiltered_snr[start_idx:stop_idx, mic]) > new_peak_thresh(start_idx, stop_idx));
        for row=1:size(high_snr_bounds, 1)
            high_snr_locations[high_snr_bounds[row, 1]:high_snr_bounds[row, 2], mic] .= 1;
        end
    end
    return high_snr_locations;
end

"""
    findroughchirpsequencebounds(snr::AbstractArray, mic::Int,
        signal_thresh::Number, peak_thresh::Number, maxfilter_length::Int)
                                                            -> Matrix{Int}

For a single microphone/channel, converts the output of `findhighsnrregions` to
a two-column matrix, where the first column is the start index of each
presumed chirp sequence.

Inputs:
- `snr`: estimated SNR of the audio data, produced by `estimatesnr` (full 
    matrix).
- `signal_thresh`, `peak_thresh`, `snr_drop_thresh`,
    `peak_snr_thresh_radius`: thresholds, described in `findhighsnrregions`.
- `maxfilter_length`: half-length, in samples, of the maximum filter.

Output: described above.
"""
function findhighsnrregionidxs(snr::AbstractArray, mic::Int, 
        signal_thresh::Number, peak_thresh::Number, maxfilter_length::Int;
        snr_drop_thresh=20, peak_snr_thresh_radius=2000) :: Matrix{Int}

    snr = vectortomatrix(snr);
    high_snr_locations = findhighsnrregions(snr[:, mic], signal_thresh, peak_thresh, 
        maxfilter_length; snr_drop_thresh=snr_drop_thresh,
        peak_snr_thresh_radius=peak_snr_thresh_radius);
    result = getboundsfromboxes(high_snr_locations);

    ## Adjust for the max-filtering done.
    result[:, 1] = result[:, 1] .+ maxfilter_length .- 1;
    result[:, 2] = result[:, 2] .- maxfilter_length .+ 1;
    return result;
end

"""
    adjusthighsnridxs(snr::AbstractArray, mic::Int,
        rough_bounds::Vector{Int}, max_end_idx::Int; tail_snr_thresh::Real,
        max_seq_len=MAX_SEQUENCE_LENGTH, maxfilter_seq_end=50) -> Vector{Int}
    
Given rough bounds for a single chirp sequence (one row of the output of
`findhighsnrregionidxs`), adjust the end index to ensure that the chirp
sequence isn't cut off early.

The process is similar to `findhighsnrregions`, but with more lenient
thresholds.
1. Apply a `maxfilter` to the `snr` array, with a filter size that is ideally
    longer than the one used for `findhighsnrregionidxs`.
2. Find the first index of the maxfiltered SNRs that goes below
    `tail_snr_thresh`, a threshold lower than the one used for
    `findhighsnrregionidxs`, and sets the end index of the chirp
    sequence to this. If this index is beyond `max_end_idx`, then the end of
    the chirp sequence is set to `max_end_idx`.

Inputs:
- `snr`: estimated SNR of the audio data, produced by `estimatesnr` (full 
    matrix).
- `mic`: microphone number, from 1 to 4.
- `rough_bounds`: row of the matrix produced by `findhighsnrregionidxs`.
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
function adjusthighsnridxs(snr::AbstractArray, mic::Int,
        rough_bounds::AbstractArray{Int}, max_end_idx::Int,
        tail_snr_thresh::Real; max_seq_len=MAX_SEQUENCE_LENGTH, maxfilter_length=50) :: Vector{Int}
    refined_bounds = copy(matrixtovector(rough_bounds));

    max_end_idx = min(max_end_idx, refined_bounds[1] + max_seq_len);
    
    seq_end = findfirst(maxfilter(snr[rough_bounds[2]:max_end_idx, mic], maxfilter_length) .<= tail_snr_thresh);
    seq_end = isnothing(seq_end) ? max_end_idx : rough_bounds[2] + seq_end[1] - 1;
    refined_bounds[2] = min(seq_end, max_end_idx);

    return refined_bounds;
end

"""
    getvocalizationtimems(chirp_start_index::Int, mic::Int,
        location_data::Matrix{Real}, mic_positions::Matrix{Real}; 
        buffer_time_ms=100, fs_video=360,
        interp_type=QuadraticInterpolation) -> Real

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
function getvocalizationtimems(chirp_start_index::Int, mic::Int, location_data::AbstractArray,
        mic_positions::AbstractArray; buffer_time_ms=100, fs_video=FS_VIDEO,
        interp_type=QuadraticInterpolation) :: Real

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
    groupchirpsequencesbystarttime(
        chirp_sequence_bounds_per_mic::AbstractArray{Matrix{Int}},
        snr::Matrix{Real}, y::Matrix{Real}, location_data::Matrix{Real},
        mic_locations::Matrix{Real}, single_chirp_snr_thresh=100,
        vocalization_start_tolerance_ms=1.5, any_mic_snr_thresh=45) 
                        -> Vector{Dict{Int, ChirpSequence}}, Vector{Real}

Given start and end indices of chirp sequences, for all microphones,
determine which chirp sequences came from the same initial chirp. Only keep
chirp sequences arising from vocalizations heard by at least two microphones.

Inputs:
- `chirp_sequence_bounds_per_mic`: array of `[chirp sequence indices for mic 1,
    ..., chirp sequence indices for mic 4]`. "Chirp sequence indices for
    mic i" refers to a two-column matrix where the first column is the start indices
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
- `any_mic_snr_thresh` (default: 45): if none of the microphones have an SNR
    above this threshold, then we probably picked up an echo instead of a chirp,
    which should be disregarded.
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
function groupchirpsequencesbystarttime(chirp_sequence_bounds_per_mic::AbstractArray{Matrix{Int}},
        snr::AbstractArray, y::AbstractArray, location_data::AbstractArray, mic_locations::AbstractArray;
        single_mic_snr_thresh=100, any_mic_snr_thresh=45, vocalization_start_tolerance_ms=1.5)

    snr = vectortomatrix(snr);
    y = vectortomatrix(y);
    ### Data structures to return:
    #################################################################
    # chirp_sequences: example
    # [{1 -> ChirpSequences(...), 2 -> ChirpSequence(...)},
    #  {2-> ChirpSequence(...), 4 -> ChirpSequence(...), 3 -> ChirpSequence(...)},
    #  ...]
    # Described in the docstring.
    chirp_sequences = Vector{Dict{Int, ChirpSequence}}(undef, 0);

    # vocalization_times: vector, where each element is the estimated vocalization
    # time of the corresponding chirp sequence.
    vocalization_times = Vector{Real}(undef, 0);
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
    current_seq_idx = ones(Int, n_mics);

    # number of chirp sequences detected for each microphone
    num_seqs = [size(chirp_sequence_bounds_per_mic[mic], 1) for mic=1:n_mics];
    current_vocalization_time = 0;
    current_seqs = Dict{Int, ChirpSequence}();

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
            if (length(current_seqs) >= 2 && 
                        any([maximum(seq.snr_data) > any_mic_snr_thresh for seq=values(current_seqs)])) ||
                    ((length(current_seqs) == 1) && 
                        maximum(first(values(current_seqs)).snr_data) >= single_mic_snr_thresh)
                push!(vocalization_times, current_vocalization_time);
                push!(chirp_sequences, current_seqs);
            end  
            # clear current_seqs       
            current_seqs = Dict{Int, ChirpSequence}();
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
    function plotchirpsequence(chirp_seq::Dict{Int, ChirpSequence};
        plot_separate=false, plot_spectrogram=false, same_length=true,
        n_cols=true)
    
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
- `same_length` (default: true): zero-pad shorter chirp sequences at the end
    so that all chirp sequences are the same length. Only relevant for
    `plot_separate` or `plot_spectrogram`.
- `n_cols (default: 2)`: number of plots per row.
"""
function plotchirpsequence(chirp_seq_all_mics::Dict{Int, ChirpSequence};
        plot_separate=false, plot_spectrogram=false, same_length=true, n_cols=2)
    ## For plotting separately
    n_mics = length(chirp_seq_all_mics);
    sorted_mics = sort(collect(keys(chirp_seq_all_mics)));

    retval = nothing;

    num_plots = Int(floor((n_mics + n_cols - 1) / n_cols)) * n_cols;
    plots = Matrix(undef, num_plots, 1);

    longest_seq = 0;
    for mic=sorted_mics
        seq_i = chirp_seq_all_mics[mic];
        longest_seq = max(longest_seq, seq_i.length);
    end

    for (i, mic)=enumerate(sorted_mics)
        seq_i = chirp_seq_all_mics[mic];
        plot_idxs = seq_i.start_idx:seq_i.start_idx+seq_i.length-1;

        mic_data = seq_i.mic_data;
        if same_length && (plot_separate || plot_spectrogram)
            mic_data = vcat(mic_data, zeros(longest_seq - seq_i.length));
            plot_idxs = seq_i.start_idx:seq_i.start_idx+longest_seq-1;
        end

        if plot_spectrogram
            plots[i] = plotSTFTtime(mic_data, nfft=256, noverlap=255, title=(@sprintf "Isolated chirp sequence for mic %d" mic));
        elseif plot_separate
            plots[i] = plotmicdata(mic_data,  plot_idxs=plot_idxs , title=(@sprintf "Isolated chirp sequence for mic %d" mic));
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
        return plot(plots..., layout=(Int(num_plots / n_cols), n_cols), size=(min(n_cols*700, 1500), 300*num_plots / n_cols))
    end
    return retval;
end

"""
    plotchirpsequenceboxes(start_ms::Real, stop_ms::Real,
        vocalization_times::Array, chirp_sequences::Array{Dict{Int, ChirpSequence}},
        y::Matrix{Real}, mics=1:size(y, 2))

Plots audio data (specified by matrix `y`) from `start_ms` to `stop_ms`
milliseconds, with boxes around all chirp sequences in that interval. Estimated
vocalization times are written above all boxes.

Inputs:
- `start_ms`: start time of the plot, in milliseconds.
- `stop_ms`: stop time of the plot, in milliseconds.
- `vocalization_times`: list of vocalization times output by
    `groupchirpsequencesbystarttime`.
- `chirp_sequences`: list of mappings from microphone to `ChirpSequence` object
    produced by `groupchirpsequencesbystarttime`.
- `y`: matrix of microphone data, where each column is a different microphone.
- `mics` (default: all): microphones for which to plot chirp sequences.
"""
function plotchirpsequenceboxes(start_ms::Real, stop_ms::Real, vocalization_times::AbstractArray, 
        chirp_sequences::Array{Dict{Int, ChirpSequence}}, y::AbstractArray;
        mics=1:size(y, 2), annotation_fontsize=8, plot_width=1500)

    y = vectortomatrix(y);
    start_seq_idx = findfirst(vocalization_times .>= start_ms);
    stop_seq_idx = findlast(vocalization_times .<= stop_ms);
    if start_ms > stop_ms || isnothing(start_seq_idx) || isnothing(stop_seq_idx)
        println("ERROR: no chirp sequences in the timeframe specified");
        return;
    end
    start_seq_idx = start_seq_idx[1]
    for seq_idx=start_seq_idx:-1:1
        new_start_idx = false
        seq = chirp_sequences[seq_idx]
        for mic=mics
            if haskey(chirp_sequences[seq_idx], mic) && 
                audioindextoms(seq[mic].start_idx + seq[mic].length - 1) > start_ms
                new_start_idx = true;
                start_seq_idx = seq_idx;
            end
        end
        if !new_start_idx
            break;
        end
    end            
    stop_seq_idx = stop_seq_idx[1];

    start_y_idx = Int(floor(start_ms * 250)) + 1;
    stop_y_idx = min(Int(ceil(stop_ms * 250)) + 1, size(y, 1));

    plots = Matrix(undef, length(mics), 1);
    for (mic_idx, mic)=enumerate(mics)
        box_height = maximum(y[start_y_idx:stop_y_idx, mic]);
        under_box = minimum(y[start_y_idx:stop_y_idx, mic]);
        plots[mic_idx] = plotmicdata(max(start_y_idx-1000, 1):stop_y_idx, y[:, mic], title=(@sprintf "Audio Data for Mic %d" mic),
            ylims=(2*under_box, 2*box_height), label=false);
        for i=start_seq_idx:stop_seq_idx
            if haskey(chirp_sequences[i], mic)
                seq = chirp_sequences[i][mic];
                box = zeros(seq.length+2);
                box[2:end-1] .= 1;
                box_ms = audioindextoms.((seq.start_idx-1):(seq.start_idx+seq.length));
                color = :red3
                if (i % 4 == 1)
                    color = :blue3;
                elseif (i % 4 == 2)
                    color = :purple3;
                elseif (i % 4 == 3)
                    color = :darkgreen;
                end
                plot!(box_ms, box .* box_height, color=color, linewidth=2, label=false);
                plot!(box_ms, box .* under_box, color=color, linewidth=2, label=false);
                if box_ms[end] > start_ms && box_ms[1] < stop_ms
                    txt_y = (i % 2 == start_y_idx % 2) ? box_height * 1.5 : under_box * 1.5;
                    annotate!((box_ms[1] + box_ms[end]) / 2, txt_y,
                        text((@sprintf "chirp (ms):\n%d" Int(round(vocalization_times[i]))), annotation_fontsize, color));
                end
            end
        end
    end
    return plot(plots..., layout=(length(mics), 1), size=(plot_width, 300*length(mics)), xlims=(start_ms-5,stop_ms));
end