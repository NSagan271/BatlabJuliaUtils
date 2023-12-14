using DSP;

"""
    colwisefft(A::AbstractArray) -> Matrix{ComplexF64}

Applies the Fast fourier Transform (FFT) to each column of input matrix `A`.
`A` can also be a vector, in which case it is first transformed into a one-col
matrix.
"""
function colwisefft(A::AbstractArray) :: Matrix{ComplexF64}
    A = reshape(A, (size(A, 1), :));
    return mapslices(x -> DSP.fft(x), A, dims=1);
end

"""
    colwiseifft(A::AbstractArray) -> Matrix{ComplexF64}

Applies the Inverse FFT to each column of input matrix `A`.
`A` can also be a vector, in which case it is first transformed into a one-col
matrix.
"""
function colwiseifft(A::AbstractArray) :: Matrix{ComplexF64}
    A = reshape(A, (size(A, 1), :));
    return mapslices(x -> DSP.ifft(x), A, dims=1);
end

"""
    rowwisefft(A::Matrix) -> Matrix{ComplexF64}

Applies the Fast fourier Transform (FFT) to each row of input matrix `A`.
"""
function rowwisefft(A::Matrix) :: Matrix{ComplexF64}
    return mapslices(x -> DSP.fft(x), A, dims=2);
end

"""
    rowwiseifft(A::Matrix) -> Matrix{ComplexF64}

Applies the Inverse FFT to each row of input matrix `A`.
"""
function rowwiseifft(A::Matrix) :: Matrix{ComplexF64}
    return mapslices(x -> DSP.ifft(x), A, dims=2);
end