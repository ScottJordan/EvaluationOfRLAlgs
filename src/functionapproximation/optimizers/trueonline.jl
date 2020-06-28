using LinearAlgebra

abstract type AbstractTrueOnlineTraceOptimizer <: AbstractTraceOptimizer end

mutable struct TrueOnlineTraceOptimizer{T, TF} <: AbstractTrueOnlineTraceOptimizer where {T<:Real,TF<:AbstractFuncApprox}
    f::TF
    α::T
    γ::T
    λ::T
    e::Array{T}
    g::Array{T}
    Δ::Array{T}
    function TrueOnlineTraceOptimizer(fun::AbstractFuncApprox, α::T, γ::T, λ::T) where {T<:Real}
        tp = eltype(get_params(fun))
        s = size(get_params(fun))
        new{tp, typeof(fun)}(fun, tp(α), tp(γ), tp(λ), zeros(tp, s), zeros(tp, s), zeros(tp, s))
    end
end

function update!(opt::TrueOnlineTraceOptimizer{T}, delta::T, ϕ::Array{T}) where {T<:Real}
    opt.e .= (opt.γ * opt.λ) .* opt.e .+ opt.α .* ϕ .- (opt.α * opt.γ * opt.λ * dot(vec(opt.e), vec(ϕ))) .* ϕ
    if !isnan(delta)
        opt.g .= opt.e .* delta .- (opt.α * dot(vec(opt.Δ), ϕ)) .* ϕ
        opt.Δ .= opt.g
        add_to_params!(opt.f, opt.g)
    end
end

function new_episode!(opt::AbstractTrueOnlineTraceOptimizer)
    fill!(opt.e, 0.)
    fill!(opt.g, 0.)
    fill!(opt.Δ, 0.)
end

function clone(opt::TrueOnlineTraceOptimizer)
    opt2 = TrueOnlineTraceOptimizer(opt.f, opt.α, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    opt2.g .= deepcopy(opt.g)
    return opt2
end

function clone(opt::TrueOnlineTraceOptimizer, f::AbstractFuncApprox)
    opt2 = TrueOnlineTraceOptimizer(f, opt.α, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    opt2.g .= deepcopy(opt.g)
    return opt2
end
