"""
    randint(end_idx::Int64) -> Int64

Returns a random integer from 1 to `end_idx`, inclusive.
"""
randint = x::Int64 -> Int64(floor(rand() * x) + 1);

"""
    distancefrommic(location_data::Vector{Float64},
        mic_positions::Matrix{Float64}, mic_num::Int64) -> Float64

Given one frame of centroid data, compute the distance from mic `mic_num`.

Inputs:
- `location_data`: one frame of centroid data, as a vector.
- `mic_positions`: matrix of microphone positions, where each row is a
    different microphone and the columns represent coordinates (x, y, z).
- `mic_num`: microphone for which to compute distances.

Output: distance from mic `mic_num`.
"""
function distancefrommic(location_data::Vector{Float64}, mic_positions::Matrix{Float64}, mic_num::Int64) :: Float64
    return sqrt(sum((location_data .- mic_positions[mic_num, :]) .^ 2));
end

"""
    distancefrommic(location_data::Matrix{Float64},
        mic_positions::Matrix{Float64}, mic_num::Int64) -> Vector{Float64}

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
function distancefrommic(location_data::Matrix{Float64}, mic_positions::Matrix{Float64}, mic_num::Int64) :: Vector{Float64}
    return sqrt.(reshape(sum((location_data .- mic_positions[mic_num:mic_num, :]) .^ 2; dims=2), :));
end