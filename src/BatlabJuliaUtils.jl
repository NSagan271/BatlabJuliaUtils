module BatlabJuliaUtils

export colwisefft,colwiseifft, rowwisefft, rowwiseifft

export audioindextosec, audioindextoms, fftindextofrequency, getfftfrequencies, videoindextosec, sectovideoindex, getvideoslicefromtimes, getvideodataslicefromaudioindices

export STFTwithdefaults, plotSTFTdb, plotSTFT, plotSTFTtime

export readmicdata

export movingaveragefilter, maxfilter, bandpassfilter, bandpassfilterFFT, bandpassfilterspecgram, circconv, deconvolve

export getnoisesampleidxs, windowedenergy, estimatesnr

export getplottingsettings, plotmicdata, plotfftmag, myplot, myplot!, plotmicdata!

export ChirpSequence, getboundsfromboxes, findhighsnrregions, findroughchirpsequenceidxs, adjustsequenceidxs, getvocalizationtimems, groupchirpsequencesbystarttime, plotchirpsequence, plotchirpsequenceboxes

export FS, FS_VIDEO, SPEED_OF_SOUND, DEFAULT_PLOT_DIM, MAX_SEQUENCE_LENGTH

export findmelody, findmelodyhertz, getharmonic, estimatechirpbounds, getchirpstartandendindices, plotmelody, plotmelodydb, estimatechirp, plotestimatedchirps, computemelodyoffsets, plotoffsetchirps, separatechirpkwargs

export colwisenormalize, getchirpsequenceY, getinitialconditionsnr, getinitialconditionsparsity, getmelodyregularization, optimizePALM

export randint, distancefrommic

include("Defaults.jl");
include("ColWiseFFTs.jl");
include("IndexToTimeOrFrequency.jl");
include("ReadMicData.jl");
include("STFTHelpers.jl");
include("Filters.jl");
include("SNR.jl");
include("Plotting.jl");
include("ChirpSequences.jl");
include("Melody.jl");
include("Optimization.jl");
include("Misc.jl");

end # module BatlabJuliaUtils
