
using LinearAlgebra


# Fourier Types {Parameter type, full or half flag, Differentiable Flag}
mutable struct ConcatBasis{T,TB1,TB2} <: AbstractFuncApprox where {T<:Real,TB1,TB2}
    ϕ1::TB1
    ϕ2::TB2

    function ConcatBasis(::Type{T}, ϕ1, ϕ2) where {T}
        new{T,typeof(ϕ1),typeof(ϕ2)}(clone(ϕ1), clone(ϕ2))
    end
end



function call(ϕ::ConcatBasis{T}, x) where {T<:Real}
    N = get_num_outputs(ϕ)
    N1 = get_num_outputs(ϕ.ϕ1)
    feats = zeros(T, N)
    call!(view(feats, 1:N1), ϕ.ϕ1, x)
    # call!(view(feats, N1:N), ϕ.ϕ2, x)

    return feats
end

function call!(out, ϕ::ConcatBasis{T}, x) where {T<:Real}
    N = get_num_outputs(ϕ)
    N1 = get_num_outputs(ϕ.ϕ1)
    call!(view(out, 1:N1), ϕ.ϕ1, x)
    # call!(view(out, N1:N), ϕ.ϕ2, x)
end


function get_num_outputs(ϕ::ConcatBasis{T})::Int where {T}
    return get_num_outputs(ϕ.ϕ1) + get_num_outputs(ϕ.ϕ2)
end

function get_output_range(ϕ::ConcatBasis{T}) where {T}
    N = get_num_outputs(ϕ)
    N1 = get_num_outputs(ϕ.ϕ1)
    ranges = zeros(T, (N, 2))
    ranges[1:N1, :] .= get_output_range(ϕ.ϕ1)
    ranges[N1:N, :] .= get_output_range(ϕ.ϕ2)

    return ranges
end

function clone(ϕ::ConcatBasis{T})::ConcatBasis{T} where {T<:Real}
    ϕ2 = ConcatBasis(T, ϕ.ϕ1, ϕ.ϕ2)
    return ϕ2
end
