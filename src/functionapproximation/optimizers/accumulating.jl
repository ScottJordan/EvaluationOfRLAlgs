using LinearAlgebra

abstract type AbstractAccumulatingTraceOptimizer <: AbstractTraceOptimizer end

mutable struct AccumulatingTraceOptimizer{T, TF} <: AbstractAccumulatingTraceOptimizer where {T<:Real,TF<:AbstractFuncApprox}
    f::TF
    α::T
    γ::T
    λ::T
    e::Array{T}
    g::Array{T}
    function AccumulatingTraceOptimizer(fun::AbstractFuncApprox, α::T, γ::T, λ::T) where {T<:Real}
        tp = eltype(get_params(fun))
        s = size(get_params(fun))
        new{tp, typeof(fun)}(fun, tp(α), tp(γ), tp(λ), zeros(tp, s), zeros(tp, s))
    end
end

function update!(opt::AccumulatingTraceOptimizer{T}, delta::T, ϕ::Array{T}) where {T<:Real}
    @. opt.e = (opt.γ * opt.λ) * opt.e + ϕ
    if !isnan(delta)
        @. opt.g = opt.e * (delta * opt.α)
        add_to_params!(opt.f, opt.g)
    end
end

function new_episode!(opt::T, rng::TR=nothing) where {T<:AbstractAccumulatingTraceOptimizer, TR<:Union{Nothing,AbstractRNG}}
    fill!(opt.e, 0.)
end

function clone(opt::AccumulatingTraceOptimizer)
    opt2 = AccumulatingTraceOptimizer(opt.f, opt.α, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    return opt2
end

function clone(opt::AccumulatingTraceOptimizer, f::AbstractFuncApprox)
    opt2 = AccumulatingTraceOptimizer(f, opt.α, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    return opt2
end

mutable struct Parl2AccumulatingTraceOptimizer{T, TF} <: AbstractAccumulatingTraceOptimizer where {T<:Real,TF<:AbstractFuncApprox}
    f::TF
    α::T
    γ::T
    λ::T
    e::Array{T}
    g::Array{T}
    function Parl2AccumulatingTraceOptimizer(fun::AbstractFuncApprox, γ::T, λ::T) where {T<:Real}
        tp = eltype(get_params(fun))
        s = size(get_params(fun))
        new{tp, typeof(fun)}(fun, 1., tp(γ), tp(λ), zeros(tp, s), zeros(tp, s))
    end
end

function update!(opt::Parl2AccumulatingTraceOptimizer{T}, delta::T, ϕ::Array{T}, ϕp::Array{T}) where {T<:Real}
    @. opt.e = (opt.γ * opt.λ) * opt.e + ϕ
    d = dot(vec(opt.e), vec(opt.γ .* ϕp .- ϕ))
    if d < 0.
        opt.α = min(opt.α, -1 / d)
    end
    if !isnan(delta)
        @. opt.g = opt.e * (delta * opt.α)
        add_to_params!(opt.f, opt.g)
    end
end


# modified Parl2 for policy gradient uses dot between e and phi
function update!(opt::Parl2AccumulatingTraceOptimizer{T}, delta::T, ϕ::Array{T}) where {T<:Real}
    @. opt.e = (opt.γ * opt.λ) * opt.e + ϕ
    d = dot(vec(opt.e), vec(-ϕ))
    if d < 0.
        opt.α = min(opt.α, -1. / d)
    end
    if !isnan(delta)
        @. opt.g = opt.e * (delta * opt.α)
        add_to_params!(opt.f, opt.g)
    end
end

function clone(opt::Parl2AccumulatingTraceOptimizer)
    opt2 = Parl2AccumulatingTraceOptimizer(opt.f, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    opt2.α = opt.α
    return opt2
end

function clone(opt::Parl2AccumulatingTraceOptimizer, f::AbstractFuncApprox)
    opt2 = Parl2AccumulatingTraceOptimizer(f, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    opt2.α = opt.α
    return opt2
end
