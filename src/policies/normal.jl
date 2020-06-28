function sample_normal!(z, μ, σ, rng)
    @. z = μ + σ * randn((rng,), eltype(μ))
end

function logpdf_normal(z, μ, σ)
    logp = -log(sqrt(2.0 * π)) - (z - μ)^2 / (2.0 * σ^2) - log(σ)
    return logp
end

# something might be wrong with the gradient of the log probability calculation. 
mutable struct LinearNormalPolicy{TP, TB, TS} <: AbstractPolicy where {TP<:Real,TB<:AbstractFuncApprox,TS<:Bool}
    θ::Array{TP}
    ϕ::TB
    σ::Array{TP, 1}
    μ::Array{TP,1}
    action::Array{TP, 1}
    feats::Array{TP, 1}

    function LinearNormalPolicy(::Type{T}, state_dim::Int, num_actions::Int, sigma, train_sigma::Bool=true) where {T}
        θmu = vec(zeros(T,state_dim, num_actions))
        σ = ones(T, num_actions) .* sigma
        ϕ = IdentityBasis(state_dim)
        if train_sigma
            θ = cat(θmu, σ, dims=1)
        else
            θ = θmu
        end

        a = zeros(tp, num_actions)
        μ = zeros(tp, num_actions)
        feats = zeros(T, get_num_outputs(ϕ))
        new{T,typeof(ϕ),train_sigma}(θ, ϕ, σ, μ, a, feats)
    end

    function LinearNormalPolicy(ϕ::AbstractFuncApprox, num_actions::Int, sigma, train_sigma::Bool=true)
        tp = eltype(get_params(ϕ))
        θmu = vec(zeros(tp,get_num_outputs(ϕ), num_actions))
        σ = ones(tp, num_actions) .* sigma
        if train_sigma
            θ = cat(θmu, σ, dims=1)
        else
            θ = θmu
        end

        a = zeros(tp, num_actions)
        μ = zeros(tp, num_actions)
        feats = zeros(tp, get_num_outputs(ϕ))
        new{tp,typeof(ϕ),train_sigma}(θ, clone(ϕ), σ, μ, a, feats)
    end
end

function get_num_params(π::LinearNormalPolicy)::Int
    return length(π.θ)
end

function get_thetamu(π::LinearNormalPolicy{T,TB,false}) where {T,TB}
    N = get_num_outputs(π.ϕ)
    return reshape(π.θ, N, :)
end

function get_thetamu(π::LinearNormalPolicy{T,TB,true}) where {T,TB}
    N = get_num_outputs(π.ϕ)
    A = length(π.σ)
    return reshape(view(π.θ, 1:N*A), N, A)
end

function call(fun::LinearNormalPolicy{TP,IdentityBasis}, x::Int)::Array{TP,1} where {TP<:Real}
    θ = get_thetamu(fun)
    return θ[x, :]
end

function call!(out::Array{TP, 1}, fun::LinearNormalPolicy{TP,IdentityBasis}, x::Int) where {TP<:Real}
    θ = get_thetamu(fun)
    out .= θ[x, :]
end

function call(fun::LinearNormalPolicy{TP,IdentityBasis}, x::Int, y::Int)::TP where{TP<:Real}
    θ = get_thetamu(fun)
    return θ[x, y]
end

function call(fun::LinearNormalPolicy{TP,IdentityBasis}, x::Array{TP,1})::Array{TP,1} where {TP<:Real}
    θ = get_thetamu(fun)
    return θ'*x
end

function call!(out::Array{TP, 1}, fun::LinearNormalPolicy{TP,IdentityBasis}, x::Array{TP,1}) where {TP<:Real}
    θ = get_thetamu(fun)
    out .= θ'*x
end

function call(fun::LinearNormalPolicy{TP}, x::Array{TP,1})::Array{TP,1} where {TP<:Real}
    call!(fun.feats, fun.ϕ, x)
    θ = get_thetamu(fun)
    return θ'*fun.feats
end

function call!(out::Array{TP, 1}, fun::LinearNormalPolicy{TP}, x::Array{TP,1}) where {TP<:Real}
    call!(fun.feats, fun.ϕ, x)
    θ = get_thetamu(fun)
    out .= θ'*fun.feats
end

function call(fun::LinearNormalPolicy{TP,IdentityBasis}, x::Array{TP,1}, y::Int)::TP where {TP<:Real}
    θ = get_thetamu(fun)
    return dot(θ[:, y],x)
end

function call(fun::LinearNormalPolicy{TP}, x::Array{TP,1}, y::Int)::TP where {TP<:Real}
    call!(fun.feats, fun.ϕ, x)
    θ = get_thetamu(fun)
    return dot(θ[:, y], fun.feats)
end

function get_mean!(π::LinearNormalPolicy, x)
    call!(π.μ,π,x)
end

function get_mean(π::LinearNormalPolicy{T}, x)::Array{T,1} where {T<:Real}
    get_mean!(π, x)
    return π.μ
end

function get_action!(π::LinearNormalPolicy{T}, x, rng::AbstractRNG)::T where {T<:Real}
    get_mean!(π, x)
    sample_normal!(π.action, π.μ, π.σ, rng)
    logp = sum(logpdf_normal.(π.action, π.μ, π.σ))
    return logp
end

function grad_mu!(grad, gmu, x)
    grad .= x*gmu'
end

function grad_mu!(grad, gmu, x::Int)
    grad[x, :] .= gmu
end

function grad_std!(grad, amu, std)
    @. grad = (-1 + (amu / std)^2) / std
end

function gradient_logp!(grad::Array{T}, π::LinearNormalPolicy{T, IdentityBasis,true}, x, action)::T where {T<:Real}
    fill!(grad, 0.)
    θmu = get_thetamu(π)
    num_theta = length(θmu)
    gtheta = reshape(view(grad, 1:num_theta), size(θmu))
    get_mean!(π, x)
    std = π.σ

    amu = @. (action - π.μ)
    gmu = @. amu / std^2

    grad_mu!(gtheta, gmu, x)
    grad_std!(view(grad, num_theta+1:length(grad)), amu, std)

    logp = sum(logpdf_normal.(π.action, π.μ, std))  # TODO make this not redundant computation

    return logp
end

function gradient_logp!(grad::Array{T}, π::LinearNormalPolicy{T, IdentityBasis,false}, x, action)::T where {T<:Real}
    fill!(grad, 0.)
    num_theta = get_num_params(π)
    gtheta = reshape(view(grad, 1:num_theta), size(get_thetamu(π)))
    get_mean!(π, x)
    std = π.σ

    amu = @. (action - π.μ)
    gmu = @. amu / std^2

    grad_mu!(gtheta, gmu, x)

    logp = sum(logpdf_normal.(π.action, π.μ, std))

    return logp
end

function gradient_logp!(grad::Array{T}, π::LinearNormalPolicy{T,TB,true}, x, action::Int)::T where {T<:Real,TB}
    fill!(grad, 0.)
    θmu = get_thetamu(π)
    num_theta = length(θmu)
    gtheta = reshape(view(grad, 1:num_theta), size(θmu))
    get_mean!(π, x)
    std = π.σ

    amu = @. (action - π.μ)
    gmu = @. amu / std^2

    grad_mu!(gtheta, gmu, π.feats)
    grad_std!(view(grad, num_theta+1:length(grad)), amu, std)

    logp = sum(logpdf_normal.(π.action, π.μ, std))

    return logp
end

function gradient_logp!(grad::Array{T}, π::LinearNormalPolicy{T,TB,false}, x, action::Int)::T where {T<:Real,TB}
    fill!(grad, 0.)
    num_theta = get_num_params(π)
    gtheta = reshape(view(grad, 1:num_theta), size(get_thetamu(π)))
    get_mean!(π, x)
    std = π.σ

    amu = @. (action - π.μ)
    gmu = @. amu / std^2

    grad_mu!(gtheta, gmu, π.feats)

    logp = sum(logpdf_normal.(π.action, π.μ, std))

    return logp
end

function get_action_gradient_logp!(grad::Array{T}, π::LinearNormalPolicy{T, IdentityBasis,true}, x, rng::AbstractRNG)::T where {T<:Real}
    fill!(grad, 0.)
    θmu = get_thetamu(π)
    num_theta = length(θmu)
    gtheta = reshape(view(grad, 1:num_theta), size(θmu))

    get_mean!(π, x)
    sample_normal!(π.action, π.μ, π.σ, rng)
    logp = sum(logpdf_normal.(π.action, π.μ, std))

    std = π.σ

    amu = @. (action - π.μ)
    gmu = @. amu / std^2

    grad_mu!(gtheta, gmu, x)
    grad_std!(view(grad, num_theta+1:length(grad)), amu, std)

    return logp
end

function get_action_gradient_logp!(grad::Array{T}, π::LinearNormalPolicy{T, IdentityBasis,false}, x, rng::AbstractRNG)::T where {T<:Real}
    fill!(grad, 0.)
    θmu = get_thetamu(π)
    num_theta = length(θmu)
    gtheta = reshape(view(grad, 1:num_theta), size(θmu))
    get_mean!(π, x)
    sample_normal!(π.action, π.μ, π.σ, rng)
    logp = sum(logpdf_normal.(π.action, π.μ, std))

    std = π.σ

    amu = @. (action - π.μ)
    gmu = @. amu / std^2

    grad_mu!(gtheta, gmu, x)

    return logp
end

function get_action_gradient_logp!(grad::Array{T}, π::LinearNormalPolicy{T, TB,true}, x, rng::AbstractRNG)::T where {T<:Real,TB}
    fill!(grad, 0.)
    θmu = get_thetamu(π)
    num_theta = length(θmu)
    gtheta = reshape(view(grad, 1:num_theta), size(θmu))
    get_mean!(π, x)
    sample_normal!(π.action, π.μ, π.σ, rng)
    std = π.σ
    logp = sum(logpdf_normal.(π.action, π.μ, std))



    amu = @. (π.action - π.μ)
    gmu = @. amu / std^2

    grad_mu!(gtheta, gmu, π.feats)
    grad_std!(view(grad, num_theta+1:length(grad)), amu, std)

    return logp
end

function get_action_gradient_logp!(grad::Array{T}, π::LinearNormalPolicy{T, TB,false}, x, rng::AbstractRNG)::T where {T<:Real,TB}
    fill!(grad, 0.)
    θmu = get_thetamu(π)
    num_theta = length(θmu)
    gtheta = reshape(view(grad, 1:num_theta), size(θmu))
    get_mean!(π, x)
    sample_normal!(π.action, π.μ, π.σ, rng)
    std = π.σ
    logp = sum(logpdf_normal.(π.action, π.μ, std))



    amu = @. (π.action - π.μ)
    gmu = @. amu / std^2

    grad_mu!(gtheta, gmu, π.feats)

    return logp
end


function set_params!(π::LinearNormalPolicy{T,TB,true}, θ::Array{T}) where {T,TB}
    π.θ .= θ
    clamp!(view(π.θ, length(π.θ)-length(π.σ)+1:length(π.θ)), 0.001, 100)
    π.σ .= π.θ[end-length(π.σ)+1:end]
end

function set_params!(π::LinearNormalPolicy{T,TB,false}, θ::Array{T}) where {T,TB}
    π.θ .= θ
end

function get_params(π::LinearNormalPolicy)
    π.θ
end

function copy_params!(params::Array{T}, π::LinearNormalPolicy{T}) where {T}
    vec(params) .= vec(π.θ)
end

function copy_params(π::LinearSoftmaxPolicy{T})::Array{T} where {T}
    return deepcopy(π.θ)
end

function add_to_params!(π::LinearNormalPolicy, Δθ)
    @. π.θ += Δθ
end

function add_to_params!(π::LinearNormalPolicy{T,TB,true}, Δθ) where {T,TB}
    @. π.θ += Δθ
    clamp!(view(π.θ,length(π.θ)-length(π.σ)+1:length(π.θ)), 0.001, 100)
    π.σ .= π.θ[end-length(π.σ)+1:end]
end

function clone(π::LinearNormalPolicy{T, TB, TS})::LinearNormalPolicy{T,TB,TS} where {T,TB,TS}
    A = length(π.σ)
    π₂ = LinearNormalPolicy(π.ϕ, A, π.σ, TS)
    π₂.action = deepcopy(π.action)
    π₂.μ = deepcopy(π.μ)
    return π₂
end
