
mutable struct RandomRBFBasis{T,TF} <: AbstractFuncApprox where {T<:Real,TF<:Bool}
    C::Array{T, 2}
    range_low::Array{T, 1}
    range_width::Array{T, 1}
    function RandomRBFBasis(ranges::Array{T, 2}, num_basis, normalize::Bool, rng::AbstractRNG) where {T<:Real}
        tp = eltype(ranges)
        C = sample_centers(T, size(ranges)[1], num_basis, rng)
        new{tp,normalize}(C, ranges[:, 1], ranges[:, 2] - ranges[:, 1])
    end
    RandomRBFBasis(C::Array{T, 2}, range_low::Vector{T}, range_width::Vector{T}, normalize::Bool) where {T} = new{T,normalize}(deepcopy(C), deepcopy(range_low), deepcopy(range_width))
end

function sample_centers(T::Type, num_inputs::Int, num_centers::Int, rng::AbstractRNG)::Array{T, 2}
    C = (rand(rng, T, (num_centers, num_inputs)) .- 0.5)
    C .= C ./ maximum(abs.(C), dims=2)
    C .*= rand(rng, T, num_centers) .* 0.5
    return C
end


function processinputs!(feats, ϕ::RandomRBFBasis{T}, x::Array{T,1}) where {T<:Real}
    fill!(feats, 0.)
    for i in 1:size(ϕ.C)[2]
        y = (x[i] - ϕ.range_low[i]) / ϕ.range_width[i]
        feats .+= (view(ϕ.C, :, i) .- y).^2
    end
    @. feats = sqrt(feats)
    @. feats = exp(feats)
end

function processinputs!(feats, ϕ::RandomRBFBasis{T,true}, x::Array{T,1}) where {T<:Real}
    fill!(feats, 0.)
    for i in 1:size(ϕ.C)[2]
        y = (x[i] - ϕ.range_low[i]) / ϕ.range_width[i]
        feats .+= (view(ϕ.C, :, i) .- y).^2
    end
    @. feats = sqrt(feats)
    @. feats = exp(feats)
    feats ./= sum(feats)
end

function call(ϕ::RandomRBFBasis{T}, x::Array{T, 1})::Array{T,1} where {T<:Real}
    N = get_num_outputs(ϕ)
    feats = zeros(T, N)
    processinputs!(view(feats, 2:N), ϕ, x)
    feats[1] = 1.
    return feats
end

function call!(out, ϕ::RandomRBFBasis{T}, x::Array{T, 1}) where {T<:Real}
    N = get_num_outputs(ϕ)
    processinputs!(view(out, 2:N), ϕ, x)
    out[1] = 1.
end

function get_params(ϕ::RandomRBFBasis)
    return ϕ.C
end

function copy_params!(params::Array{T}, ϕ::RandomRBFBasis{T}) where {T<:Real}
    p = reshape(params, size(ϕ.C))
    copyto!(p, ϕ.C)
end

function copy_params(ϕ::RandomRBFBasis{T}) where {T<:Real}
    params = deepcopy(ϕ.C)
    return params
end

function set_params!(ϕ::RandomRBFBasis{T}, θ::Array{T}) where {T<:Real}
    ϕ.C .= reshape(θ,size(ϕ.C))
end

function add_to_params!(ϕ::RandomRBFBasis{T}, θ::Array{T}) where {T<:Real}
    ϕ.C .+= reshape(θ, size(ϕ.C))
end

function get_num_params(ϕ::RandomRBFBasis)::Int64
    return length(ϕ.C)
end

function get_num_outputs(ϕ::RandomRBFBasis{T})::Int where {T}
    return size(ϕ.C)[1] + 1
end

function get_output_range(ϕ::RandomRBFBasis{T}) where {T}
    ranges = zeros(T, (get_num_outputs(ϕ), 2))
    ranges[:, 2] .= 1.
    #ranges[1, 1]  = 1.  # first term is always bias
    return ranges
end

function clone(ϕ::RandomRBFBasis{T, TF})::RandomRBFBasis{T, TF} where {T<:Real,TF}
    ϕ2 = RandomRBFBasis(ϕ.C, ϕ.range_low, ϕ.range_width, TF)
    return ϕ2
end
