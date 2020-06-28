
struct IdentityBasis <: AbstractFuncApprox
    num_inputs
    state_ranges
    IdentityBasis(num_inputs) = new(num_inputs, nothing)
    IdentityBasis(state_ranges::Array{T,2}) where {T} = new(size(state_ranges)[1], state_ranges)
end

function get_num_outputs(f::IdentityBasis)
    return f.num_inputs
end

function get_output_range(f::IdentityBasis)
    if f.state_ranges == nothing
        ranges = zeros((f.num_inputs, 2))
        ranges[:, 1] .= -Inf
        ranges[:, 2] .= Inf
        return ranges
    else
        return f.state_ranges
    end
end

function call(f::IdentityBasis, x::T)::T where T
    return x
end

function call!(out, f::IdentityBasis, x)
    out .= x
end

function clone(f::IdentityBasis)::IdentityBasis
    return IdentityBasis(f.num_inputs)
end

export IdentityBasis



include("fourierbasis.jl")
include("randomrbf.jl")
include("concat.jl")
export FourierBasis, RandomRBFBasis, ConcatBasis
