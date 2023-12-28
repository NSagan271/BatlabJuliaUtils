using MAT;

"""
    readmicdata(mat_filename::String, n_channels=4) -> Matrix{Real}

Reads audio data from a MAT file into a matrix where each column represents a 
different microphone.

These can be produced by running `matlab_utils/tdms_to_mat.m` on a TDMS file 
with fields including  `Voltage_0`, `Voltage_1`, etc.

Inputs:
- `mat_filename`: name of a `.mat` file with variables `Voltage_i`, where
    microphone index i counts up from 0, each variable is a time-series array 
    of voltages for the corresponding microphone, and all arrays are of the 
    same length.
- `n_channels` (default 4): number of microphones

Output:
- `y`: N x K matrix, where N is the number of samples per microphone and K
     is the number of microphones. Each column corresponds to a different
    microphone.
"""
function readmicdata(mat_filename::String, n_channels=4) :: Matrix
    file = matopen(mat_filename);
     
    y = Matrix(undef, 0, 0);
    for i=0:(n_channels-1)
        yi = read(file, "Voltage_" * string(i))'; 
        if i == 0
            y = yi;
        else
            y = hcat(y, yi);
        end
    end
    close(file);
    return y;
end
