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

```@docs
ChirpSequence
getboundsfromboxes
findhighsnrregions
findroughchirpsequenceidxs
adjustsequenceidxs
getvocalizationtimems
groupchirpsequencesbystarttime
plotchirpsequence
```

# "Melody"
We define the "melody" of the vocalization as the fundamental harmonic of the chirp.

This section contains documentation for estimating the melody, as well as some basic methods for separating chirps from echos.

```@docs
findmelody
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

# Misc
```@docs
randint
distancefrommic
```