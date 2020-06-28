using LinearAlgebra

# Not sure if this is implemented correctly or is just sensitive to meta learning rate.
mutable struct TIDBDOptimizer{T, TF} <: AbstractAccumulatingTraceOptimizer where {T<:Real,TF<:AbstractFuncApprox}
    f::TF
    η::T         # meta stepsize
    α::Array{T}  # learning rates, i.e., exp.(β)
    β::Array{T}  # ln learning rates
    γ::T         # discount factor
    λ::T         # eligibility trace decay
    e::Array{T}  # eligibility trace vector
    h::Array{T}  # memory vector for stepsizes
    g::Array{T}  # gradient place holder

    function TIDBDOptimizer(fun::AbstractFuncApprox, b::T, η::T, γ::T, λ::T) where {T<:Real}
        tp = eltype(get_params(fun))
        s = size(get_params(fun))

        β = zeros(tp, s)
        h = zeros(tp, s)
        fill!(β, b)
        α = exp.(β)
        new{tp, typeof(fun)}(fun, η, α, β, tp(γ), tp(λ), zeros(tp, s), h, zeros(tp, s))
    end
end

function update!(opt::TIDBDOptimizer{T}, δ::T, ϕ::Array{T}) where {T<:Real}
    @. opt.β += opt.η * δ * ϕ * opt.h
    @. opt.α = exp(opt.β)
    @. opt.e = (opt.γ * opt.λ) * opt.e + ϕ

    @. opt.g = opt.e * (δ * opt.α)
    add_to_params!(opt.f, opt.g)
    @. opt.h = opt.h * clamp(1. - opt.α * ϕ * opt.e, 0., Inf) + opt.α * δ * opt.e
end


function clone(opt::TIDBDOptimizer)
    opt2 = TIDBDOptimizer(opt.f, 1.0, opt.η, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    opt2.β .= deepcopy(opt.β)
    opt2.α .= deepcopy(opt.α)
    opt2.h .= deepcopy(opt.h)
    return opt2
end

function clone(opt::TIDBDOptimizer, f::AbstractFuncApprox)
    opt2 = TIDBDOptimizer(f, 1.0, opt.η, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    opt2.β .= deepcopy(opt.β)
    opt2.α .= deepcopy(opt.α)
    opt2.h .= deepcopy(opt.h)
    return opt2
end


mutable struct AutoTIDBDOptimizer{T, TF} <: AbstractAccumulatingTraceOptimizer where {T<:Real,TF<:AbstractFuncApprox}
    f::TF
    η::T         # meta stepsize
    α::Array{T}  # learning rates, i.e., exp.(β)
    β::Array{T}  # ln learning rates
    γ::T         # discount factor
    λ::T         # eligibility trace decay
    e::Array{T}  # eligibility trace vector, z in the paper
    h::Array{T}  # memory vector for stepsizes
    g::Array{T}  # gradient place holder
    z::Array{T}  # max observed update trace, η in the paper
    τ::T         # large value to perform normalization of meta learning rate

    function AutoTIDBDOptimizer(fun::AbstractFuncApprox, b::T, η::T, τ::T, γ::T, λ::T) where {T<:Real}
        tp = eltype(get_params(fun))
        s = size(get_params(fun))

        β = zeros(tp, s)
        h = zeros(tp, s)
        z = zeros(tp, s)
        fill!(β, b)
        α = exp.(β)
        new{tp, typeof(fun)}(fun, η, α, β, tp(γ), tp(λ), zeros(tp, s), h, zeros(tp, s), z, τ)
    end
end

function update!(opt::AutoTIDBDOptimizer{T}, δ::T, ϕ::Array{T}, ϕ′::Array{T}) where {T<:Real}
    gpp = (opt.γ .* ϕ′ .- ϕ)
    p1 = δ .* gpp .* opt.h
    p2 = opt.z .- (1. / opt.τ) .* opt.α .* gpp .* opt.e .* (δ .* ϕ .* opt.h .- opt.z)
    opt.z .= max.(abs.(p1), p2)

    @. opt.β -= opt.η / opt.z * δ * gpp * opt.h #* ϕ * opt.h
    M = max.(exp.(opt.β) * dot(vec(gpp), vec(opt.e)), 1.)
    @. opt.β -= log.(M)
    @. opt.α = exp(opt.β)
    @. opt.e = (opt.γ * opt.λ) * opt.e + ϕ

    @. opt.g = opt.e * (δ * opt.α)
    add_to_params!(opt.f, opt.g)
    # @. opt.h = opt.h * clamp(1. - opt.α * ϕ * opt.e, 0., Inf) + opt.α * δ * opt.e
    @. opt.h = opt.h * clamp(1. + opt.α * gpp * opt.e, 0., Inf) + opt.α * δ * opt.e
end


function clone(opt::AutoTIDBDOptimizer)
    opt2 = AutoTIDBDOptimizer(opt.f, 1.0, opt.η, opt.τ, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    opt2.β .= deepcopy(opt.β)
    opt2.α .= deepcopy(opt.α)
    opt2.h .= deepcopy(opt.h)
    opt2.z .= deepcopy(opt.z)
    return opt2
end

function clone(opt::AutoTIDBDOptimizer, f::AbstractFuncApprox)
    opt2 = AutoTIDBDOptimizer(f, 1.0, opt.η, opt.τ, opt.γ, opt.λ)
    opt2.e .= deepcopy(opt.e)
    opt2.β .= deepcopy(opt.β)
    opt2.α .= deepcopy(opt.α)
    opt2.h .= deepcopy(opt.h)
    opt2.z .= deepcopy(opt.z)

    return opt2
end
