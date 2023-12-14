using Printf;
using LinearAlgebra;

"""
    colwisenormalize(Y::Matrix) -> Matrix

Normalizes each column of matrix `Y` such that each column has an amplitude
(i.e., maximum absolute value) of 1.

Inputs:
- `Y`: matrix, where each column is a different channel of data.

Outputs:
- `Y`, with each column divided by its amplitude.
""" 
function colwisenormalize(Y::Matrix)
    return Y ./ maximum(abs.(Y); dims=1);
end

"""
    getchirpsequenceY(chirp_seq_all_mics::Dict{Int64, ChirpSequence},
        offsets::Dict{Int64, Int64}, pad_len::Int64; normalize=true)
                                        -> Matrix{Float64}, Vector{Int64}

Given chirp sequence objects (for all microphones) corresponding to the same
initial vocalization, produce a matrix of microphone data, where each column is
a different microphone. The microphone data is zero-padded at the beginning and
the end by `pad_len` zeros. In addition, if the beginning of any chirp was cut
off (as determined by `offsets`, which is the output of
`computemelodyoffsets`), the beginning is zero-padded by the number of samples
that were cut off. By default, each column of the output is normalized to have
amplitude 1.

Inputs:
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- `offsets`: output of `computemelodyoffsets`; mapping of microphone index to
    how many samples were cut off (if any) at the beginning of the chirp.
- `pad_len`: how many zeros to add to the beginning and end of the mic data.
    This is used to mitigate edge effects from circular convolution.
- `normalize` (default: `true`): whether to ensure each column of the output
    has amplitude `1`.

Outputs:
- `Y`: L x K matrix, where L is the length of the longest chirp sequene in
    `chirp_seq_all_mics`, plus `2*pad_len` and `K` is the number of
    microphones used.
- `mics`: list of microphone number, where the microphone number at each index
    is the microphone used for the corresponding column of `Y`.
"""
function getchirpsequenceY(chirp_seq_all_mics::Dict{Int64, ChirpSequence}, offsets::Dict{Int64, Int64}, pad_len::Int64; normalize=true)
    max_length = 1;

    for mic=keys(chirp_seq_all_mics)
        max_length = max(max_length, chirp_seq_all_mics[mic].length + offsets[mic]);
    end

    Y = zeros(max_length + 2*pad_len, length(chirp_seq_all_mics))
    mics = sort(collect(keys(chirp_seq_all_mics)));
    for (i, mic)=enumerate(mics)
        if offsets[mic] >= 0
            Y[pad_len+1+offsets[mic]:chirp_seq_all_mics[mic].length+pad_len+offsets[mic], i] = chirp_seq_all_mics[mic].mic_data;
        else
            Y[pad_len+1:chirp_seq_all_mics[mic].length+offsets[mic]+pad_len, i] = chirp_seq_all_mics[mic].mic_data[1-offsets[mic]:end];
        end
    end

    if normalize
        Y = colwisenormalize(Y);
    end
    
    return Y, mics;
end

"""
    getchirpsequenceY(chirp_seq_all_mics::Dict{Int64, ChirpSequence},
        peak_snr_thresh::Real, pad_len::Int64; normalize=true,
        offset_kwargs...) -> Matrix{Float64}, Vector{Int64}

Like the above function (`getchirpsequenceY(chirp_seq_all_mics, offsets,
pad_len, normalize)`), except the `offsets` are automatically computed using
`computemelodyoffsets`.

Inputs:
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `pad_len`: how many zeros to add to the beginning and end of the mic data.
    This is used to mitigate edge effects from circular convolution.
- `normalize` (default: `true`): whether to ensure each column of the output
    has amplitude `1`.
- `offset_kwargs`: optionally, you can pass in any keyword arguments for
    `computemelodyoffsets`.

Outputs:
- `Y`: L x K matrix, where L is the length of the longest chirp sequene in
    `chirp_seq_all_mics`, plus `2*pad_len` and `K` is the number of
    microphones used.
- `mics`: list of microphone number, where the microphone number at each index
    is the microphone used for the corresponding column of `Y`.
"""
function getchirpsequenceY(chirp_seq_all_mics::Dict{Int64, ChirpSequence}, peak_snr_thresh::Real, pad_len::Int64; normalize=true, offset_kwargs...)
    offsets = computemelodyoffsets(chirp_seq_all_mics, peak_snr_thresh; offset_kwargs...);
    return getchirpsequenceY(chirp_seq_all_mics, offsets, pad_len, normalize=true);
end

"""
    getchirpsequenceY(chirp_seq_all_mics::Dict{Int64, ChirpSequence},
        pad_len::Int64; normalize=true) -> Matrix{Float64}, Vector{Int64}

Like the above function (`getchirpsequenceY(chirp_seq_all_mics, offsets,
pad_len, normalize)`), except the `offsets` are all set to zero.

Inputs
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- `pad_len`: how many zeros to add to the beginning and end of the mic data.
    This is used to mitigate edge effects from circular convolution.
- `normalize` (default: `true`): whether to ensure each column of the output
    has amplitude `1`.

Outputs:
- `Y`: L x K matrix, where L is the length of the longest chirp sequene in
    `chirp_seq_all_mics`, plus `2*pad_len` and `K` is the number of
    microphones used.
- `mics`: list of microphone number, where the microphone number at each index
    is the microphone used for the corresponding column of `Y`.
"""
function getchirpsequenceY(chirp_sechirp_seq_all_micsquence::Dict{Int64, ChirpSequence}, pad_len::Int64; normalize=true)
    offsets = Dict{Int64, Int64}();
     for mic=keys(chirp_seq_all_mics)
        offsets[mic] = 0;
    end
    return getchirpsequenceY(chirp_seq_all_mics, offsets, pad_len, normalize=true);
end

"""
    getinitialconditionsparsity(Y::Matrix{Float64},
        chirp_seq_all_mics::Dict{Int64, ChirpSequence}, mics::Vector{Int64},
        peak_snr_thresh::Real; chirp_kwargs...) 
                                    -> Vector{Float64}, Matrix{Float64}, Int64

Produces an initial condition for the blind deconvolution algorithm (i.e.: an
estimate of the bat chirp and the impulse responses mapping the bat chirp to
the audio output of each microphone). It uses the chirp estimate (from 
`estimatechirp`) for one of the microphones as the estimate of the bat chirp,
and performs Fourier deconvolution (see the `deconvolve` function) to get the
impulse responses. The chirp is estimated using whichever microphone produces
the sparserst impulse response.

This is not recommended for noisy data; `getinitialconditionsnr` is better for
that case.

Inputs:
- `Y`: matrix of audio data, produced by `getchirpsequenceY`.
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- `mics`: list of microphones used, produced by `getchirpsequenceY`.
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `chirp_kwargs`: you can pass in any keywords for `estimatechirp`.

Outputs: 
- `X_init`: estimated bat vocalization.
- `H_init`: matrix of estimated impulse responses, the k-th column of `H_init`,
    convolved with `X_init`, produces the k-th column of `Y`.
- `longest_chirp`: length of the longest estimated chirp.
"""
function getinitialconditionsparsity(Y::Matrix{Float64}, chirp_seq_all_mics::Dict{Int64, ChirpSequence}, 
        mics::Vector{Int64}, peak_snr_thresh::Real; chirp_kwargs...)
    best_X_init = nothing;
    best_cost = Inf;
    longest_chirp = 0;

    for (i,mic)=enumerate(mics)
        start_idx, X_init = estimatechirp(chirp_seq_all_mics[mic], peak_snr_thresh; chirp_kwargs...);
        X_init_L = length(X_init);
        # normalize
        X_init /= maximum(abs.(X_init));
        
        Y_col = Y[:, i];
        X_init = vcat(X_init, zeros(size(Y, 1) - length(X_init)));
        

        H_approx = deconvolve(Y, X_init);

        cost = norm(H_approx, 1);
        
        longest_chirp = max(X_init_L, longest_chirp);
        if cost < best_cost
            best_X_init = X_init;
            best_cost = cost;
        end
    end
    X_init = vcat(best_X_init, zeros(size(Y, 1) - size(best_X_init, 1)));
    H_init = deconvolve(Y, X_init);
    return X_init, H_init, longest_chirp;
end

"""
    getinitialconditionsparsity(Y::Matrix{Float64},
        chirp_seq_all_mics::Dict{Int64, ChirpSequence}, mics::Vector{Int64},
        peak_snr_thresh::Real; chirp_kwargs...) 
                                    -> Vector{Float64}, Matrix{Float64}, Int64

Produces an initial condition for the blind deconvolution algorithm (i.e.: an
estimate of the bat chirp and the impulse responses mapping the bat chirp to
the audio output of each microphone). It uses the chirp estimate (from 
`estimatechirp`) for the highest-SNR microphone as the estimate of the bat chirp,
and performs Fourier deconvolution (see the `deconvolve` function) to get the
impulse responses.

This is the preferred method of obtaining the initial condition for noisy data.

Inputs:
- `Y`: matrix of audio data, produced by `getchirpsequenceY`.
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- `mics`: list of microphones used, produced by `getchirpsequenceY`.
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `chirp_kwargs`: you can pass in any keywords for `estimatechirp`.

Outputs: 
- `X_init`: estimated bat vocalization.
- `H_init`: matrix of estimated impulse responses, the k-th column of `H_init`,
    convolved with `X_init`, produces the k-th column of `Y`.
- `longest_chirp`: length of the longest estimated chirp.
"""
function getinitialconditionsnr(Y::Matrix{Float64}, chirp_seq_all_mics::Dict{Int64, ChirpSequence}, 
        mics::Vector{Int64}, peak_snr_thresh::Real; chirp_kwargs...)
    longest_chirp = 0;
    best_snr = -Inf;
    best_snr_mic = 0;
    for mic=mics
        snr = maximum(chirp_seq_all_mics[mic].snr_data);
        if snr > best_snr
            best_snr = snr;
            best_snr_mic = mic;
        end

        start_idx, X_init = estimatechirp(chirp_seq_all_mics[mic], peak_snr_thresh; chirp_kwargs...);
        X_init_L = length(X_init);
        longest_chirp = max(X_init_L, longest_chirp);
    end
    _, X_init = estimatechirp(chirp_seq_all_mics[best_snr_mic], peak_snr_thresh; chirp_kwargs...);
    X_init /= maximum(abs.(X_init));

    X_init = vcat(X_init, zeros(size(Y, 1) - size(X_init, 1)));
    H_init = deconvolve(Y, X_init);
    return X_init, H_init, longest_chirp;
end

"""
function getmelodyregularization(chirp_seq_all_mics::Dict{Int64, ChirpSequence},
    N::Int64, peak_snr_thresh::Real; melody_radius = 2, nfft=256,
    stft_stride=Int64(floor(nfft/4)), max_freq_hz=100_000, chirp_kwargs...)
                                                                    -> Matrix

Produces a weight matrix (of the same dimensions of the STFT of a bat
vocalization, where the length of the vocalization is length of the longest
estimated chirp for `chirp_seq_all_mics`) that quantifies how far any part
of the spectrogram is from the melody or one of its harmonics.

Details of the algorithm are as follows:
1. Initialize `Mx2`, the element-wise square of the weight matrix, to be all
    infinity.
2. For each microphone in `chirp_seq_all_mics`:
    a. Find the melody, chirp start/end indices, and whether the beginning of
        the chirp was cut off.
    b. Loop through all harmonics under `max_freq_hz`:
        i. Find the distance of each point on the spectrogram to some range
            around the melody. The radius of this range is either `err_radius`
            or the slope of the melody at the given point, whichever is larger.

            This provides some leeway in where the actual melody falls, where the leeway is determined by how fast the melody is changing.
        ii. Set `Mx2` to the element-wise minimum of its current value and the
            squared distance from step i.
3. Truncate `Mx2` to the maximum estimated chirp length.
4. Downsample `Mx2` by a factor of `stft_stride`, setting each index of the
    downsampled version to the minimum value in a time radius of `nfft/2`.

Inputs:
- `chirp_seq_all_mics`: mapping of microphone index to `ChirpSequence` object.
- `N`: upper bound on the length of the bat vocalization.
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `melody_radius` (default: 2): described in step 2.b.i above.
- `nfft` (default: 256): window length to use for the STFT.
- `stft_stride` (default: `nfft/4`): amount to downsample in step 4 above.
- `max_freq_hz` (default: 100k): used to find the highest possible harmonic.
- `chirp_kwargs`: optionally, you can pass in any keyward arguments for
    `findmelody`, `estimatechirpbounds`, or `computemelodyoffsets`.

Outputs:
- `Mx2`: weight matrix, described above.
- `max_chirp_len`: maximum chirp length from `estimatechirp`.
"""
function getmelodyregularization(chirp_seq_all_mics::Dict{Int64, ChirpSequence}, N::Int64,
    peak_snr_thresh::Real; melody_radius = 2, nfft=256, stft_stride=Int64(floor(nfft/4)),
    max_freq_hz=100_000, chirp_kwargs...)

    # Get keyword arguments for findmelody, estimatechirpbounds,
    # and computemelodyoffsets
    chirp_kwargs = merge(Dict{Symbol, Any}(:nfft => nfft), chirp_kwargs...);
    melody_kwargs, chirp_bound_kwargs, offset_kwargs = separatechirpkwargs(;chirp_kwargs...);

    # Find the peak SNR of each microphone (for this chirp sequence )
    snrs = Dict{Int64, Float64}();
    for (mic, seq_one_mic)=pairs(chirp_seq_all_mics)
        snrs[mic] = maximum(seq_one_mic.snr_data);
    end
    max_snr = maximum(values(snrs));

    # N_f is the height of the spectrogram
    N_f = Int64(nfft / 2 + 1);
    M_f_idxs = [1:N_f;];

    # this will be populated with the maximum chirp length
    max_chirp_len = 0;

    # Weight matrix to return. Initialize 
    Mx2 = ones(N_f, N) * Inf;

    offsets = computemelodyoffsets(chirp_seq_all_mics, peak_snr_thresh; chirp_kwargs...);
        
    for (mic, seq_one_mic)=pairs(chirp_seq_all_mics)
        # lower-SNR mics should matter less...
        snr_weight = snrs[mic] / max_snr;
        
        # Find the melody and chirp start/end
        melody = findmelody(seq_one_mic, peak_snr_thresh; melody_kwargs...);
        chirp_start, chirp_end = estimatechirpbounds(seq_one_mic, melody, peak_snr_thresh; chirp_bound_kwargs...);

        chirp_len = chirp_end - chirp_start + 1;

        melody = melody[chirp_start:chirp_end];

        # If the beginning of the melody was cut off, or there was extraneous
        # noise at the beginning of the chirp, adjust the melody and/or chirp
        # length accordingly.
        offset = offsets[mic];
        if offset < 0
            melody = melody[-offset+1:end];
            chirp_len += offset;
            offset = 0;
        else
            chirp_len += offset;
        end
        
        max_chirp_len = max(chirp_len, max_chirp_len);
        
        # loop through all valid harmonics
        harmonic = 1;
        while fftindextofrequency(harmonic*minimum(melody), nfft) < max_freq_hz
            harmonic_melody = melody;
            if harmonic > 1
                harmonic_melody = getharmonic(seq_one_mic, melody, harmonic);
            end

            # We want to find the distance of each point on the spectrogram to
            # some range around the melody. The radius of this range is either
            # `err_radius` or the slope of the melody at the given point,
            # whichever is larger.
            err = 1.0 .* abs.(M_f_idxs .- transpose(harmonic_melody));
            err_radius = max.(transpose(abs.(harmonic_melody[1:end-2] - harmonic_melody[3:end]) / 2), melody_radius)
            err[:, 2:end-1] = max.(err[:, 2:end-1], err_radius) .- err_radius;

            #  Set `Mx2` to the element-wise minimum of its current value and the
            # squared distance from step i.
            Mx2[:, offset+1:chirp_len] = min.(Mx2[:, offset+1:chirp_len], (err / snr_weight) .^ 2);
            harmonic += 1;
        end
    end
    Mx2 = Mx2[:, 1:max_chirp_len];
    # downsample Mx2 without losing too much information
    start_idx = Int64(nfft/2);
    window_idx = 1;

    Mx2_downsampled = Mx2;
    while start_idx <= size(Mx2, 2);
        Mx2_downsampled[:, window_idx] = minimum(Mx2[:, start_idx-Int64(nfft/2)+1:min(size(Mx2, 2), start_idx+Int64(nfft/2))]; dims=2);
        start_idx += stft_stride;
        window_idx += 1
    end
    Mx2_downsampled = Mx2_downsampled[:, 1:window_idx-1];
    Mx2_downsampled /= (N_f * size(Mx2, 2));
    Mx2_downsampled /= 10;
    return Mx2_downsampled, max_chirp_len;
end


"""
function optimizePALM(chirp_seq::Dict{Int64, ChirpSequence}, Y::Matrix{Float64}, 
    H_init::Matrix{Float64}, X_init::Array{Float64}, peak_snr_thresh::Real,
    data_fitting_weight::Real, sparsity_weight::Real, melody_weight::Real;
    max_iter=1000, alpha=1e-3, gamma_H=1, gamma_X=1, num_debug=10, nfft=256,
    stft_stride=Int64(floor(nfft/4)), chirp_kwargs...) -> Matrix, Matrix, Int64
                                                
Performs blind deconvolution, formulated as an optimization problem with a
combination of the following objectives:
1. Data-fitting: making sure `X`, convolved with any column of `H`, the matrix
    of impulse responses, is close to the corresponding column of `Y`, the
    microphone data marix produced by `getchirpsequenceY`.
2. Impulse response sparsity: encouraging sparsity of `H`, measured by the
    absolute sum (L1 norm) of the elements of `H`.
3. Melody following: making sure the energy of the spectrogram  of `X` is close
    to the melody or one of its harmonics. See `getmelodyregularization` for
    more details.

This optimization problem is solved using PALM (proximal alternating linearized
optimization: https://arxiv.org/abs/1604.00526), following the general process
outlined in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7730569/.

Inputs:
- `chirp_seq`: mapping of microphone index to `ChirpSequence` object.
- `Y`: matrix of microphone data; output of `getchirpsequenceY`.
- `H_init`: initial condition for the impulse responses; output of
    `getinitialconditionsnr` or `getinitialconditionsparsity`.
- `X_init`: initial condition for the bat vocalization. Produced by the same
    function as `H_init`.
- `peak_snr_thresh`: threshold set for the peak SNR of a chirp sequence.
- `data_fitting_weight`: weight of the data-fitting term in the optimization
    objective. Can be thought of as a percentage of how much that term
    matters (60 is a generally good value).
- `sparsity_weight`: weight of the sparsity term in the objective. 10 is a
    generally good value.
- `melody_weight`: weight of the melody-following term in the objective. 50 is
    a generally good value.
- `max_iter` (default: 1000): number of iterations to run the algorithm for.
- `alpha` (default: 0.001): the absolute sum of the elements of `H` is
    approximated using the smooth function `|x| ≈ sqrt(x^2 + alpha^2) - alpha`,
    where `alpha` is some small number
- `gamma_H` (default: 1): factor to slow down updates of `H` (must be at least
    1, or the algorithm might not converge).
- `gamma_X` (default: 1): factor to slow down updates of `X` (must be at least
    1, or the algorithm might not converge).
- `num_debug` (default: 10): number of times to print debug statements over the
    course of the optimization algorithm.
- `nfft` (default: 256): window length to use for the spectrogram of `X`.
- `stft_stride` (default: `nfft/4`): spacing between adjacent spectrogram
    windows.
- `chirp_kwargs`: optionally, you can pass in any keyward arguments for
    `findmelody`, `estimatechirpbounds`, or `computemelodyoffsets`.

Outputs:
- `X`: bat vocalization, as estimated by the optimization algorithm,
- `H`: matrix of impulse responses, as estimated by the optimization algorithm.
- `max_chirp_len`: maximum chirp length from `estimatechirp`.
"""
function optimizePALM(chirp_seq::Dict{Int64, ChirpSequence}, Y::Matrix{Float64}, 
        H_init::Matrix{Float64}, X_init::Array{Float64}, peak_snr_thresh::Real,
        data_fitting_weight::Real, sparsity_weight::Real, melody_weight::Real;
        max_iter=1000, alpha=1e-3, gamma_H=1, gamma_X=1, 
        num_debug=10, nfft=256, stft_stride=Int64(floor(nfft/4)), 
        chirp_kwargs...)

    mu_1, mu_2, mu_3 = data_fitting_weight, sparsity_weight, melody_weight

    # print a debug statement after this many iterations
    debug_freq = ceil(max_iter / num_debug);

    # sizes and optimization variables
    N, K = size(Y);
    H = copy(H_init);
    X = copy(X_init);

    # Take the FFT of all signals
    Y_fft = colwisefft(Y) ./ sqrt(N);
    X_fft = colwisefft(X) ./ sqrt(N);
    H_fft = colwisefft(H) ./ sqrt(N);

    # Get the weight matrix for regularization on the STFT of X, which ensures
    # that the energy of the spectrogram is close to the melody or one of its
    # harmonics
    Mx2, max_chirp_len = getmelodyregularization(
        chirp_seq, N, peak_snr_thresh;
        nfft=nfft, stft_stride=stft_stride,
        chirp_kwargs...);

    
    # Window function used to get the spectrogram of X
    W = hamming(nfft);
    N_f, N_w = size(Mx2);
    Mx2 *= mu_3;

    # Lipschitz constant used to calculate gradient descent step size
    Lr = N_w * maximum(W) .^ 2 * maximum(Mx2);

    ## Function for printing debugging statements
    function debug(iter=0)
        lstsq_term = 0;
        for k=1:K
            lstsq_term += norm(vec(circconv(X, H) .- Y), 2) ^ 2;
        end
        lstsq_term /= (N*K);
        l1 = norm(vec(H), 1) ./ K;

        if iter == 0
            @printf "[Initial] Average squared error = %0.4e; H sparsity loss = %f\n" lstsq_term l1;
        else
            @printf "[Iter %d] Average squared error = %0.4e; H sparsity loss = %f\n" iter lstsq_term l1;
        end
        flush(stdout);
    end
    debug(); # Print a debug statement for the initial condition

    # Loop until we reach max_iter
    for iter=1:max_iter
        ##### Update H

        # Each element of C_diags is the gradient descent step size for the
        # corresponding row of H
        L1_prime = mu_1 / K^2 * maximum(abs.(X_fft) .^ 2);
        C_diags = gamma_H .* (L1_prime .+ mu_2 / (N*K) ./ sqrt.(H .^ 2 .+ alpha ^ 2));

        # Gradient of the objective function with respect to H
        grad_H_f = mu_1 / K^2 * real.(colwiseifft(conj.(X_fft) .* (sqrt(N) .* X_fft .* H_fft .- Y_fft))) + mu_2 / (N*K) .* H ./ sqrt.(H .^ 2 .+ alpha ^ 2);

        # Perform a gradient descent step on H
        H = H .- grad_H_f ./ C_diags;

        # Take the FFT of the updated H
        H_fft = colwisefft(H) ./ sqrt(N);

        ##### Update X

        # Gradient descent step size for updating X
        L2 = mu_1 / K^2 * maximum((abs.(H_fft) .^ 2) * ones(K)) + Lr;

        # Gradient of the data-fitting term with respect to X
        grad_X_f = mu_1 / K^2 * real.(
            DSP.ifft((sqrt(N) .* (abs.(H_fft) .^ 2) * ones(K)) .* X_fft .- (conj.(H_fft) .* Y_fft) * ones(K))
        )[1:max_chirp_len];

        # Compute the gradient of the melody-following regularization term with
        # respect to X
        grad_X_r = zeros(max_chirp_len);
        for m=1:N_w # loop through thw windows
            win_start = (m-1)*stft_stride+1;

            # Mx2, times the Fourier matrix, times the window function, times X
            Mx2FWX = zeros(ComplexF64, nfft);
            Mx2FWX[1:N_f] = Mx2[:, m] .* DSP.fft(W .* X[win_start:win_start+nfft-1])[1:N_f];

            # Use conjugate symmetry of the Fourier transform
            Mx2FWX[end:-1:Int64(nfft/2+2)] = conj.(Mx2FWX[2:Int64(nfft/2)]);

            # Take the inverse Fourier transform
            ifft_Mx2FWX = DSP.ifft(Mx2FWX);

            len = min(win_start+nfft-1, max_chirp_len) - win_start + 1;
            grad_X_r[win_start:win_start+len-1] =
                    grad_X_r[win_start:win_start+len-1] .+ (W .* real.(ifft_Mx2FWX))[1:len];
        end

        # Gradient of the full objective function with respect to X
        grad_X_f = grad_X_f .+ grad_X_r;
        d = gamma_X * L2;
        
        # Gradient descent update for X; only updating until max_chirp_len
        # (the rest of X is forced to be zero).
        X[1:max_chirp_len] = X[1:max_chirp_len] .- 1/d .* grad_X_f;

        # Take the FFT of the updated X
        X_fft = colwisefft(X) ./ sqrt(N);

        if iter % debug_freq == 0
            debug(iter);
        end
    end

    return X, H, max_chirp_len;
end