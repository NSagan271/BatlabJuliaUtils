include("Defaults.jl");

"""
    audioindextosec(index::Int64; fs=250 kHz) -> Float64

Given an index of the audio data, calculate the number of seconds since the
beginning of the audio data.

Inputs:
- `index`: index of audio data
- `fs`: sampling frequency, in Hertz. Default set in Defaults.jl.

Output:
- Seconds since the beginning of the audio data
"""
function audioindextosec(index::Int64; fs=FS) :: Float64
    return index / fs;
end

"""
    audioindextoms(index::Int64; fs=250 kHz) -> Float64

Given an index of the audio data, calculate the number of milliseconds
since the beginning of the audio data.

Inputs:
- `index`: index of audio data
- `fs`: sampling frequency, in Hertz. Default set in Defaults.jl.

Output:
- Milliseconds since the beginning of the audio data
"""
function audioindextoms(index::Int64; fs=FS) :: Float64
    return index / fs * 1000;
end

"""
    fftindextofrequency(index::Int64, N_fft::Int64; fs=250 kHz) -> Float64

Given an index of an Discrete Fourier Transform taken on a segment of audio
datal determine what frequency (in Hz) it corresponds to.

Inputs:
- `index`: index of the Fourier Transform.
- `N_fft`: length of the Fourier Transform, in samples.
- `fs`: sampling frequency, in Hertz. Default set in Defaults.jl.

Output:
- Frequency, in Hz, of the specified Fourier Transform index
"""
function fftindextofrequency(index::Int64, N_fft::Int64; fs=FS) :: Float64
    omega = 2*pi/N_fft * (index-1);
    if omega > pi
        omega = 2pi - omega;
    end
    return omega/(2pi) * fs;
end

"""
    getfftfrequencies(N_fft::Int64; fs=250 kHz) -> Array{Float64}

Return an array of length `N_fft`, where each element is the frequency, in
Hz, of the corresponding index of a length-`N_fft` Fourier Transform of
a segment of audio data.

Inputs:
- `N_fft`: length of the Fourier Transform, in samples.
- `fs`: sampling frequency, in Hertz. Default set in Defaults.jl.
"""
function getfftfrequencies(N_fft::Int64; fs=FS) :: Array{Float64}
    return fftindextofrequency.(1:N_fft, N_fft; fs=fs);
end

"""
    videoindextosec(idx_video::Int64, L_video::Int64; fs_video=360) -> Float64

Given an index of the video data, return the number of seconds since the
start of the audio data.

The video and audio data are synchronized such that the end of the video
data is simultaneous with the 8-second mark of the audio data.

Inputs:
- `idx_video`: index of the video data.
- `L_video`: length, in frames, of the video data.
- `fs_video`: sampling rate of the video data, in Hertz. Default set in 
    Defaults.jl.
"""
function videoindextosec(idx_video::Int64, L_video::Int64; fs_video=FS_VIDEO) :: Float64
    return (-(L_video - idx_video) / fs_video + 8);
end

"""
    sectovideoindex(time_audio::Number, L_video::Int64; fs_video=360) -> Int64

Given a time (since the start of the audio data) in seconds, calculate the
index of the closest video frame.

The video and audio data are synchronized such that the end of the video
data is simultaneous with the 8-second mark of the audio data.

Inputs:
- `time_audio`: time, in seconds, since the start of the audio data.
- `L_video`: length, in frames, of the video data.
- `fs_video`: sampling rate of the video data, in hertz. Default set in 
    Defaults.jl.
"""
function sectovideoindex(time_audio::Number, L_video::Int64; fs_video=FS_VIDEO) :: Int64
    return Int64(round(L_video - (8 - time_audio) * fs_video));
end

"""
    getvideoslicefromtimes(location_data::Matrix{Float64}, t1::Float64,
        t2::Float64, fs_video=360) -> UnitRange{Int64}, Matrix{Float64}

Returns the video data corresponding to time interval `[t1, t2]` of the audio
data, where `t1` and `t2` are in seconds.

The two datasets are synchronized as follows: the end of the video data
coincides with the 8-second mark of the audio data.

Inputs:
- `location_data`: full centroid data, where the columns represent coordinates
    (x, y, z).
- `t1`: start time (inclusive) of interval, in seconds since the onset of the
    audio data.
- `t2`: end time (inclusive) of interval, in seconds since the onset of the
    audio data.
- `fs_video` (default set in `Defaults.jl`): sampling frequency of the centroid
    data, in Hertz.

Outputs:
- `slice_idxs`: indices of the audio data corresponding to the `[t1, t2]` slice
    taken.
- `centroid_slice`: centroid data in the interval `[t1, t2]`.
"""
function getvideoslicefromtimes(location_data::Matrix{Float64}, t1::Float64, t2::Float64, fs_video=FS_VIDEO)
    # if t1_ms > 8000 || t2_ms > 8000
    #     @warn "Trying to access video data past the end. Data returned will only be up to the 8 second mark.";
    #     t1_ms = min(t1_ms, 8000);
    #     t2_ms = min(t2_ms, 8000);
    # end
    t1 = max(t1, 0);
    t2 = max(t2, 0);

    L = size(location_data, 1);

    t1_idx = sectovideoindex(t1, L);
    t2_idx = sectovideoindex(t2, L);

    return t1_idx:t2_idx, location_data[t1_idx:t2_idx, :];
end

"""
    getvideodataslicefromaudioindices(location_data::Matrix{Float64},
        t1_idx::Float64, t2_idx::Float64, fs_video=360, FS=250k) 
                                        -> UnitRange{Int64}, Matrix{Float64}

Same as `getvideoslicefromtimes`, but takes audio data indices in lieu of times.

Inputs:
- `location_data`: full centroid data, where the columns represent coordinates
    (x, y, z).
- `t1_idx`: index of the audio data (inclusive) at which to start the centroid
    data slice.
- `t2_idx`: index of the audio data (inclusive) at which to end the centroid
    data slice.
- `fs_video` (default set in `Defaults.jl`): sampling frequency of the centroid
    data, in Hertz.
- `fs` (default set in `Defaults.jl`): sampling frequency of the audio
    data, in Hertz.

Outputs: see `getvideoslicefromtimes`
"""
function getvideodataslicefromaudioindices(location_data::Matrix{Float64}, t1_idx::Int64, t2_idx::Int64, 
            fs_video=FS_VIDEO, fs=FS)
    return getvideoslicefromtimes(location_data, audioindextosec(t1_idx, fs=FS), 
        audioindextosec(t2_idx, fs=FS), fs_video=FS_VIDEO);
end