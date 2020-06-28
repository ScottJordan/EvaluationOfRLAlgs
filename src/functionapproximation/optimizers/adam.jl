using LinearAlgebra

mutable struct Adam{T, TF} <: AbstractOptimizer where {T<:Real,TF<:AbstractFuncApprox}
    α::T
    f::TF
    β1::T
    β2::T
    ϵ::T
    t::Int
    m::Array{T}
    v::Array{T}
    Δ::Array{T}

    function Adam(fun::AbstractFuncApprox, α::T; β1::T = 0.9, β2::T = 0.999, ϵ::T =1e-8) where {T<:Real}
        tp = eltype(get_params(fun))
        m = zeros(tp, size(get_params(fun)))
        v = zeros(tp, size(get_params(fun)))
        Δ = zeros(tp, size(get_params(fun)))
        new{tp, typeof(fun)}(tp(α), fun, β1, β2, ϵ, 0, m, v, Δ)
    end
end


function update!(opt::Adam{T}, g::Array{T}) where {T<:Real}
    opt.t += 1
    t, β1, β2, ϵ = opt.t, opt.β1, opt.β2, opt.ϵ
    @. opt.m *= β1
    @. opt.m += (1.0 - β1) * g
    @. opt.v *= β2
    @. opt.v += (1.0 - β2) * g^2
    α = opt.α * √(1.0 - β2^t) / (1.0 - β1^t)
    @. opt.Δ = α * opt.m / (√opt.v + ϵ)
    # add_to_params!(opt.f, @. α * opt.m / (√opt.v + ϵ))
    add_to_params!(opt.f, opt.Δ)
end

function new_episode!(opt::Adam, rng=nothing)
    nothing
end

function clone(opt::Adam)
    opt2 = Adam(opt.f, opt.α, β1=opt.β1, β2=opt.β2, ϵ=opt.ϵ)
    opt2.t = opt.t
    opt2.m .= deepcopy(opt.m)
    opt2.v .= deepcopy(opt.v)
    opt2.Δ .= deepcopy(opt.Δ)
    return opt2
end

function clone(opt::Adam, f::AbstractFuncApprox)
    opt2 = Adam(f, opt.α, β1=opt.β1, β2=opt.β2, ϵ=opt.ϵ)
    opt2.t = opt.t
    opt2.m .= deepcopy(opt.m)
    opt2.v .= deepcopy(opt.v)
    opt2.Δ .= deepcopy(opt.Δ)
    return opt2
end
