

abstract type GradientMode end
abstract type SemiGradient <: GradientMode end
abstract type FullGradient <: GradientMode end

export GradientMode, SemiGradient, FullGradient

include("accumulating.jl")
include("trueonline.jl")
include("tidbd.jl")

export AbstractAccumulatingTraceOptimizer, AccumulatingTraceOptimizer, Parl2AccumulatingTraceOptimizer
export TIDBDOptimizer, AutoTIDBDOptimizer
export AbstractTrueOnlineTraceOptimizer, TrueOnlineTraceOptimizer
