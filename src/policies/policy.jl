# using ..Rllib
# using ..Rllib.FuncApprox

include("softmax.jl")
include("egreedy.jl")
include("normal.jl")
export LinearSoftmaxPolicy, LinearNormalPolicy
export EpsilonGreedy
