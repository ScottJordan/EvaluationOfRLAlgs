
using LinearAlgebra


# Fourier Types {Parameter type, full or half flag, Differentiable Flag}
mutable struct FourierBasis{T,TF} <: AbstractFuncApprox where {T<:Real,TF<:Bool}
    C::Array{T, 2}
    range_low::Array{T, 1}
    range_width::Array{T, 1}
    function FourierBasis(ranges::Array{T, 2}, dorder::Int, iorder::Int, full::Bool=false) where {T<:Real}
        tp = eltype(ranges)
        new{tp,full}(make_Cmat(T, size(ranges)[1], dorder, iorder), ranges[:, 1], ranges[:, 2] - ranges[:, 1])
    end
    FourierBasis(C::Array{T, 2}, range_low::Vector{T}, range_width::Vector{T}, full::Bool) where {T} = new{T,full}(deepcopy(C), deepcopy(range_low), deepcopy(range_width))
end

function increment_counter!(counter::Vector{Int}, maxDigit::Int)
    for i in length(counter):-1:1
        counter[i] += 1
        if (counter[i] > maxDigit)
            counter[i] = 0
        else
            break
        end
    end
end

function make_Cmat(T::Type, num_inputs::Int, dorder::Int, iorder::Int)::Array{T, 2}
    iTerms = iorder * num_inputs
    dTerms = (dorder+1)^num_inputs
    oTerms = min(iorder, dorder) * num_inputs
    num_feats = iTerms + dTerms - oTerms
    C = zeros(T, (num_feats, num_inputs))
    counter = zeros(Int, num_inputs)
    termCount::Int = 1

    while termCount <= dTerms
        for i in 1:num_inputs
            C[termCount, i] = counter[i]
        end
        increment_counter!(counter, dorder)
        termCount += 1
    end
    for i in 1:num_inputs
        for j in (dorder+1):(iorder)
            C[termCount, i] = j
            termCount += 1
        end
    end
    C .*= π

    return C  # transpose(C)
end


function processinputs!(feats, ϕ::FourierBasis{T}, x::Array{T,1}) where {T<:Real}
    scaled = @. (x - ϕ.range_low) / ϕ.range_width
    # feats .= ϕ.C * scaled
    @. feats = 0.0
    for i in 1:length(x)
        feats .+= view(ϕ.C, :, i) .* scaled[i]
    end
    nothing
end

function call(ϕ::FourierBasis{T,false}, x::Array{T, 1})::Array{T,1} where {T<:Real}
    feats = zeros(T, get_num_outputs(ϕ))
    processinputs!(feats, ϕ, x)
    feats .= cos.(feats)
    return feats
end

function call!(out, ϕ::FourierBasis{T,false}, x::Array{T, 1}) where {T<:Real}
    # feats = zeros(T, get_num_outputs(ϕ))
    # processinputs!(feats, ϕ, x)
    # out .= cos.(feats)
    processinputs!(out, ϕ, x)
    @. out = cos(out)
end

function call(ϕ::FourierBasis{T,true}, x::Array{T, 1})::Array{T,1} where {T<:Real}
    feats = zeros(T, get_num_outputs(ϕ))
    call!(feats, ϕ, x)
    # numhalf= size(ϕ.C)[1]
    # tmp = zeros(T, numhalf)
    # processinputs!(tmp, ϕ, x)
    # feats[numhalf+1:end] .= sin.(tmp)
    # feats[1:numhalf] .= cos.(tmp)
    return feats
end

function call!(out, ϕ::FourierBasis{T,true}, x::Array{T, 1}) where {T<:Real}
    numhalf= size(ϕ.C)[1]
    # tmp = zeros(T, numhalf)
    # processinputs!(tmp, ϕ, x)
    # out[numhalf+1:end] .= sin.(tmp)
    # out[1:numhalf] .= cos.(tmp)
    processinputs!(view(out, 1:numhalf), ϕ, x)
    out[numhalf+1:end] .= sin.(view(out, 1:numhalf))
    out[1:numhalf] .= cos.(view(out, 1:numhalf))
end


function get_params(ϕ::FourierBasis)
    return ϕ.C
end

function copy_params!(params::Array{T}, ϕ::FourierBasis{T}) where {T<:Real}
    p = reshape(params, size(ϕ.C))
    copyto!(p, ϕ.C)
end

function copy_params(ϕ::FourierBasis{T}) where {T<:Real}
    params = deepcopy(ϕ.C)
    return params
end

function set_params!(ϕ::FourierBasis{T}, θ::Array{T}) where {T<:Real}
    ϕ.C .= reshape(θ,size(ϕ.C))
end

function add_to_params!(ϕ::FourierBasis{T}, θ::Array{T}) where {T<:Real}
    ϕ.C .+= reshape(θ, size(ϕ.C))
end

function get_num_params(ϕ::FourierBasis)::Int64
    return length(ϕ.C)
end

function get_num_outputs(ϕ::FourierBasis{T,false})::Int where {T}
    return size(ϕ.C)[1]
end

function get_num_outputs(ϕ::FourierBasis{T,true})::Int where {T}
    return 2. * size(ϕ.C)[1]
end

function get_output_range(ϕ::FourierBasis{T}) where {T}
    ranges = zeros(T, (get_num_outputs(ϕ), 2))
    ranges[:, 1] .= -1.
    ranges[:, 2] .= 1.
    #ranges[1, 1]  = 1.  # first term is always bias
    return ranges
end

function clone(ϕ::FourierBasis{T, TF})::FourierBasis{T, TF} where {T<:Real,TF}
    ϕ2 = FourierBasis(ϕ.C, ϕ.range_low, ϕ.range_width, TF)
    return ϕ2
end
