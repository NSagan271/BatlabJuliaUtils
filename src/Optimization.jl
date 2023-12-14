using Printf;
using LinearAlgebra;

function colwisenormalize(Y)
    return Y ./ maximum(abs.(Y); dims=1);
end

function getchirpsequenceY(chirp_sequence::Dict{Int64, ChirpSequence}, offsets::Dict{Int64, Int64}, pad_len::Int64; normalize=true)
    max_length = 1;

    for mic=keys(chirp_sequence)
        max_length = max(max_length, chirp_sequence[mic].length + offsets[mic]);
    end

    Y = zeros(max_length + 2*pad_len, length(chirp_sequence))
    mics = sort(collect(keys(chirp_sequence)));
    for (i, mic)=enumerate(mics)
        if offsets[mic] >= 0
            Y[pad_len+1+offsets[mic]:chirp_sequence[mic].length+pad_len+offsets[mic], i] = chirp_sequence[mic].mic_data;
        else
            Y[pad_len+1:chirp_sequence[mic].length+offsets[mic]+pad_len, i] = chirp_sequence[mic].mic_data[1-offsets[mic]:end];
        end
    end

    if normalize
        Y = colwisenormalize(Y);
    end
    
    return Y, mics;
end

function getchirpsequenceY(chirp_sequence::Dict{Int64, ChirpSequence}, peak_snr_thresh::Real, pad_len::Int64; normalize=true, offset_kwargs...)
    offsets = computemelodyoffsets(chirp_sequence, peak_snr_thresh; offset_kwargs...);
    return getchirpsequenceY(chirp_sequence, offsets, pad_len, normalize=true);
end

function getchirpsequenceY(chirp_sequence::Dict{Int64, ChirpSequence}, pad_len::Int64; normalize=true)
    offsets = Dict{Int64, Int64}();
     for mic=keys(chirp_sequence)
        offsets[mic] = 0;
    end
    return getchirpsequenceY(chirp_sequence, offsets, pad_len, normalize=true);
end

"""
    getinitialconditionsparsity(Y::Matrix{Float64},
        chirp_seq_all_mics::Dict{Int64, ChirpSequence}, mics::Vector{Int64},
        peak_snr_thresh::Real; chirp_kwargs...) 
                                    -> Vector{Float64}, Matrix{Float64}, Int64
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
function getmelodyregularization(chirp_seq::Dict{Int64, ChirpSequence}, N::Int64,
    peak_snr_thresh::Real; melody_diffuseness_percent=10,
    melody_radius = 2,
    nfft=256, stft_stride=Int64(floor(nfft/4)),
    chirp_kwargs...)
"""
function getmelodyregularization(chirp_seq::Dict{Int64, ChirpSequence}, N::Int64,
    peak_snr_thresh::Real; melody_radius = 2,
    nfft=256, stft_stride=Int64(floor(nfft/4)),
    chirp_kwargs...)

    chirp_kwargs = merge(Dict{Symbol, Any}(:nfft => nfft), chirp_kwargs...);
    melody_kwargs, chirp_bound_kwargs, offset_kwargs = separatechirpkwargs(;chirp_kwargs...);

    snrs = Dict{Int64, Float64}();
    for (mic, seq_one_mic)=pairs(chirp_seq)
        snrs[mic] = maximum(seq_one_mic.snr_data);
    end
    max_snr = maximum(values(snrs));

    N_f = Int64(nfft / 2 + 1);
    M_f_idxs = [1:N_f;];
    max_chirp_len = 0;

    Mx2 = ones(N_f, N) * Inf;

    offsets = computemelodyoffsets(chirp_seq, peak_snr_thresh; chirp_kwargs...);
        
    for (mic, seq_one_mic)=pairs(chirp_seq)
        snr_weight = snrs[mic] / max_snr;
        
        melody = findmelody(seq_one_mic, peak_snr_thresh; melody_kwargs...);
        chirp_start, chirp_end = estimatechirpbounds(seq_one_mic, melody, peak_snr_thresh; chirp_bound_kwargs...);
        chirp_len = chirp_end - chirp_start + 1;

        melody = melody[chirp_start:chirp_end];
        offset = offsets[mic];
        if offset < 0
            melody = melody[-offset+1:end];
            chirp_len += offset;
            offset = 0;
        else
            chirp_len += offset;
        end
        
        max_chirp_len = max(chirp_len, max_chirp_len);
        
        harmonic = 1;
        while fftindextofrequency(harmonic*minimum(melody), nfft) < 100_000 # hardcoded max freq
            harmonic_melody = melody;
            if harmonic > 1
                harmonic_melody = getharmonic(seq_one_mic, melody, harmonic);
            end
            err = 1.0 .* abs.(M_f_idxs .- transpose(harmonic_melody));
            err_radius = max.(transpose(abs.(harmonic_melody[1:end-2] - harmonic_melody[3:end]) / 2), melody_radius)
            err[:, 2:end-1] = max.(err[:, 2:end-1], err_radius) .- err_radius;
            Mx2[:, offset+1:chirp_len] = min.(Mx2[:, offset+1:chirp_len], (err / snr_weight) .^ 2);
            harmonic += 1;
        end
    end
    Mx2 = Mx2[:, 1:max_chirp_len];
    # downsample Mx2 without losing too much info
    start_idx = Int64(nfft/2);
    window_idx = 1;
    while start_idx <= size(Mx2, 2);
        Mx2[:, window_idx] = minimum(Mx2[:, start_idx-Int64(nfft/2)+1:min(size(Mx2, 2), start_idx+Int64(nfft/2))]; dims=2);
        start_idx += stft_stride;
        window_idx += 1
    end
    Mx2 = Mx2[:, 1:window_idx-1];
    Mx2 /= (N_f * size(Mx2, 2));
    Mx2 /= 10;
    return Mx2, max_chirp_len;
end


"""
function optimizePALM(chirp_seq::Dict{Int64, ChirpSequence}, Y::Matrix{Float64}, 
    H_init::Matrix{Float64}, X_init::Array{Float64}, peak_snr_thresh::Real,
    data_fitting_weight::Real, sparsity_weight::Real, melody_weight::Real,
    pad_len::Int;  melody_diffuseness_percent=50, max_iter=1000, alpha=1e-3, gamma_1=1, gamma_2=1, 
    num_debug=10, nfft=256, stft_stride=Int64(floor(nfft/4)), 
    chirp_kwargs...)

"""
function optimizePALM(chirp_seq::Dict{Int64, ChirpSequence}, Y::Matrix{Float64}, 
        H_init::Matrix{Float64}, X_init::Array{Float64}, peak_snr_thresh::Real,
        data_fitting_weight::Real, sparsity_weight::Real, melody_weight::Real,
        pad_len::Int; max_iter=1000, alpha=1e-3, gamma_1=1, gamma_2=1, 
        num_debug=10, nfft=256, stft_stride=Int64(floor(nfft/4)), 
        chirp_kwargs...)

    mu_1, mu_2, mu_3 = data_fitting_weight, sparsity_weight, melody_weight
    debug_freq = ceil(max_iter / num_debug);

    N, K = size(Y);
    H = copy(H_init);
    X = copy(X_init);

    Y_fft = colwisefft(Y) ./ sqrt(N);
    X_fft = colwisefft(X) ./ sqrt(N);
    H_fft = colwisefft(H) ./ sqrt(N);

    Mx2, max_chirp_len = getmelodyregularization(
        chirp_seq, N, peak_snr_thresh;
        nfft=nfft, stft_stride=stft_stride,
        chirp_kwargs...);

        
    W = hamming(nfft);
    N_f, N_w = size(Mx2);
    Mx2 *= mu_3;
    Lr = N_w * maximum(W) .^ 2 * maximum(Mx2);

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
    debug();

    for iter=1:max_iter
        ##### Update H
        L1_prime = mu_1 / K^2 * maximum(abs.(X_fft) .^ 2);
        
        C_diags = gamma_1 .* (L1_prime .+ mu_2 / (N*K) ./ sqrt.(H .^ 2 .+ alpha ^ 2));
        grad_H_f = mu_1 / K^2 * real.(colwiseifft(conj.(X_fft) .* (sqrt(N) .* X_fft .* H_fft .- Y_fft))) +
                        mu_2 / (N*K) .* H ./ sqrt.(H .^ 2 .+ alpha ^ 2);

        H = H .- grad_H_f ./ C_diags;
        H_fft = colwisefft(H) ./ sqrt(N);

        ##### Update X
        L2 = mu_1 / K^2 * maximum((abs.(H_fft) .^ 2) * ones(K)) + Lr;
        grad_X_f = mu_1 / K^2 * real.(
            DSP.ifft((sqrt(N) .* (abs.(H_fft) .^ 2) * ones(K)) .* X_fft .- (conj.(H_fft) .* Y_fft) * ones(K))
        )[1:max_chirp_len];

        grad_X_r = zeros(max_chirp_len);
        for m=1:N_w
            win_start = (m-1)*stft_stride+1;

            Mx2FWX = zeros(ComplexF64, nfft);
            Mx2FWX[1:N_f] = Mx2[:, m] .* DSP.fft(W .* X[win_start:win_start+nfft-1])[1:N_f];
            Mx2FWX[end:-1:Int64(nfft/2+2)] = conj.(Mx2FWX[2:Int64(nfft/2)]);
            ifft_Mx2FWX = DSP.ifft(Mx2FWX);

            len = min(win_start+nfft-1, max_chirp_len) - win_start + 1;
            grad_X_r[win_start:win_start+len-1] =
                    grad_X_r[win_start:win_start+len-1] .+ (W .* real.(ifft_Mx2FWX))[1:len];
        end
        grad_X_f = grad_X_f .+ grad_X_r;
        d = gamma_2 * L2;
                
        X[1:max_chirp_len] = X[1:max_chirp_len] .- 1/d .* grad_X_f;
        X_fft = colwisefft(X) ./ sqrt(N);

        if iter % debug_freq == 0
            debug(iter);
        end
    end

    return X, H, max_chirp_len;
end