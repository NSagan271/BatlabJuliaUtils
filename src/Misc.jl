"""
    randint(end_idx::Int) -> Int

Returns a random integer from 1 to `end_idx`, inclusive.
"""
randint = x::Int -> Int(floor(rand() * x) + 1);

"""
    distancefrommic(location_data::Vector{Real},
        mic_positions::Matrix{Real}, mic_num::Int) -> Real

Given one frame of centroid data, compute the distance from mic `mic_num`.

Inputs:
- `location_data`: one frame of centroid data, as a vector.
- `mic_positions`: matrix of microphone positions, where each row is a
    different microphone and the columns represent coordinates (x, y, z).
- `mic_num`: microphone for which to compute distances.

Output: distance from mic `mic_num`.
"""
function distancefrommic(location_data::Vector, mic_positions::Matrix, mic_num::Int) :: Real
    return sqrt(sum((location_data .- mic_positions[mic_num, :]) .^ 2));
end

"""
    distancefrommic(location_data::Matrix{Real},
        mic_positions::Matrix{Real}, mic_num::Int) -> Vector{Real}

Given centroid data for multiple time points, compute the distance from mic
`mic_num` (for each time point).

Inputs:
- `location_data`: slice of centroid data, where the columns represent
    coordinates (x, y, z).
- `mic_positions`: matrix of microphone positions, where each row is a
    different microphone and the columns represent coordinates (x, y, z).
- `mic_num`: microphone for which to compute distances.

Output: vector of distance from mic `mic_num`, for each row of `location_data`.
"""
function distancefrommic(location_data::Matrix, mic_positions::Matrix, mic_num::Int) :: Vector
    return sqrt.(reshape(sum((location_data .- mic_positions[mic_num:mic_num, :]) .^ 2; dims=2), :));
end

"""
    matrixtovector(mat::AbstractArray)

Given an input matrix, return the vector produced by stacking all of the
columns together. If the input is already a vector, it remains unchanged.

Inputs:
- `mat`: matrix to transform to a vector.

Output:
- `vec`: e.g., if `mat` is `[1 2 3; 4 5 6]`, vec will be `[1, 4, 2, 5, 3, 6]`.
"""
function matrixtovector(mat::AbstractArray)
    return vcat(mat...);
end

"""
    vectortomatrix(vec::AbstractArray)  

Given an input vector, return a matrix that consists of one column. If the
input is already a matrix, it remains unchanged.

Inputs:
- `vec`: vector to transform to a one-column matrix.

Output:
- `mat`: single-column matrix.
"""
function vectortomatrix(vec::AbstractArray)
    return reshape(vec, (size(vec, 1), :));
end