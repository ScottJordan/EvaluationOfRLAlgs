
include("traces.jl")
include("adam.jl")

export SGD, SGA
export Adam

mutable struct SGD{T, TF} <: AbstractOptimizer where {T<:Real,TF<:AbstractFuncApprox}
    α::T
    f::TF
    function SGD(fun::AbstractFuncApprox, α::T) where {T<:Real}
        tp = eltype(get_params(fun))
        new{tp, typeof(fun)}(eltype(get_params(fun))(α), fun)
    end
end

mutable struct SGA{T, TF} <: AbstractOptimizer where {T<:Real,TF<:AbstractFuncApprox}
    α::T
    f::TF
    function SGA(fun::AbstractFuncApprox, α::T) where {T<:Real}
        tp = eltype(get_params(fun))
        new{tp, typeof(fun)}(eltype(get_params(fun))(α), fun)
    end
end

function update!(opt::SGD{T}, g::Array{T}) where{T<:Real}
    add_to_params!(opt.f, -opt.α .* g)
end

function update!(opt::SGD{T}, delta::T, g::Array{T}) where{T<:Real}
    add_to_params!(opt.f, -opt.α * delta .* g)
end

function update!(opt::SGA{T}, g::Array{T}) where{T<:Real}
    add_to_params!(opt.f, opt.α .* g)
end

function update!(opt::SGA{T}, delta::T, g::Array{T}) where{T<:Real}
    add_to_params!(opt.f, opt.α * delta .* g)
end

function new_episode!(opt::AbstractOptimizer, rng=nothing)
    return nothing
end


function clone(opt::SGD)
    opt2 = SGD(opt.f, opt.α)
end

function clone(opt::SGD, f::AbstractFuncApprox)
    opt2 = SGD(f, opt.α)
end

function clone(opt::SGA)
    opt2 = SGA(opt.f, opt.α)
end

function clone(opt::SGA, f::AbstractFuncApprox)
    opt2 = SGA(f, opt.α)
end
