# BatlabJuliaUtils Documentation

```@contents
```

# Some Julia Notes
## Keyword Arguments (`kwargs...`  Notation)
If a function is as follows:
```
function helloworld(a; kwargs)
```
then you can pass in any keyword arguments you want, as follows:
```
helloworld(2; b=3, c="hello");
```
The use of these keyword arguments, when present, is described in the documentation.

## Plotting
As Julia has some issues with plotting in Jupyter notebooks, it's recommended to use the [Plotting Helpers](@ref) listed here. If you use one of the built-in Julia plot functions, you can pass in the output of [`getplottingsettings`](@ref) as keyword arguments as follows:
```
plot(x_data, y_data; getplottingsettings("x label", "y label", "my title")...);
```
Otherwise, the plot will show up as tiny unless you pass in the keyword argument `html_output_format=:png`, and the axis labels may be cut off unless you pass in `left_margin=10Plots.mm` and `bottom_margin=10Plots.mm`.

Also, if some plotting function is not producing an output in a Jupyter notebook, make sure there is no semicolon at the end of the statement.

## Broadcasting Across an Array
To broadcast a one number-to-one number function across an array, you can add a dot (`.`) after the function name.

For instance, if you have complex array
```
A = [3.0 + 4.0im, 5.0 + 2.3im, 3.4 + 0.9im];
```
you can get the magnitude of each element by `abs.(A)`, which produces
```
1×3 Matrix{Float64}:
 5.0  5.50364  3.5171
```

## `MethodError`s
If a function throws a `MethodError: no method matching...`, where the function wants you to pass in a `Matrix` but you passed in a `Vector`, you can apply [`vectortomatrix`](@ref) to the vector before passing it into the function.
If you have the opposite problem, you can use [`matrixtovector`](@ref).

If the function takes in a single number but you want to pass in an array instead, look at [Broadcasting Across an Array](@ref).

If the datatype (e.g., `Int`, `Real`, `Float64`, etc.) of the input data is wrong, you can cast them to the correct type:
- If `A` is an array of `Int`s, you can cast them to `Float64` using `Float64.(A)`, and cast them to `Real` using `Real.(A)`.
- If `A` is an array of `Float64`s, you can cast them to `Int`s using `Int.(round.(A))` (you can replace `round` with `floor` or `ceil`, depending on how you want to round the numbers). If you omit the `round`, and `A` contains decimal numbers, then casting them to an `Int` will throw an `InexactError`.
-If `A` is an array of complex numbers but you know that the imaginary part should be zero (e.g., taking the inverse FFT of an FFT of a real array), you can do `real.(A)` to take the real part.

# Defaults.jl
`BatlabUtils/src/Defaults.jl` stores default values for sampling frequencies (250 kHz for audio and 360 Hz for video), the speed of sound (344.69 m/s), default plot dimensions, and some algorithm parameters.

# Plotting Helpers

```@docs
getplottingsettings
myplot
myplot!
plotmicdata
plotmicdata!
plotfftmag
```

# Time-Frequency Helper Functions

## ColWiseFFTs

```@docs
colwisefft
colwiseifft
rowwisefft
rowwiseifft
```

## Convert Data Indices to Time or Frequency
```@docs
audioindextosec
audioindextoms
fftindextofrequency
getfftfrequencies
videoindextosec
sectovideoindex
getvideoslicefromtimes
getvideodataslicefromaudioindices
```
## Short-Time Fourier Transforms (STFTs) and Spectrograms

```@docs
STFTwithdefaults
plotSTFTdb
plotSTFT
plotSTFTtime
```

# Read Audio Data

```@docs
readmicdata
```

# Filters

```@docs
movingaveragefilter
maxfilter
bandpassfilter
bandpassfilterFFT
bandpassfilterspecgram
circconv
deconvolve
```

# Signal-to-Noise Ratio
```@docs
getnoisesampleidxs
windowedenergy
estimatesnr
```

# Chirp Sequences

**Terminology: Chirp Sequences**

A chirp sequence is defined as all microphone outputs that result from a single bat vocalization.

This is a somewhat overloaded term, which can mean one of two things:
1. For a single microphone, a chirp and subsequent echos. We can call this a "single-mic chirp sequence".
2. The chirp and subsequent echos, but for all microphones that picked up the chirp. We can call this a "multi-mic chirp sequence".


```@docs
estimatebuzzphase
ChirpSequence
getboundsfromboxes
findhighsnrregions
findhighsnrregionidxs
adjusthighsnridxs
getvocalizationtimems
groupchirpsequencesbystarttime
plotchirpsequence
plotchirpsequenceboxes
```

# "Melody"
We define the "melody" of the vocalization as the fundamental harmonic of the chirp.

This section contains documentation for estimating the melody, as well as some basic methods for separating chirps from echos.

```@docs
findmelody
findmelodyhertz
getharmonic
smoothmelody
estimatechirpbounds
getchirpstartandendindices
plotmelody
plotmelodydb
estimatechirp
plotestimatedchirps
computemelodyoffsets
plotoffsetchirps
separatechirpkwargs
```

# Optimization
This section contains helper methods for performing blind deconvolution on a chirp sequence to estimate the initial bat vocalization.

```@docs
colwisenormalize
getchirpsequenceY
getinitialconditionsnr
getinitialconditionsparsity
getmelodyregularization
optimizePALM
```

# Miscellaneous
```@docs
matrixtovector
vectortomatrix
randint
distancefrommic
```