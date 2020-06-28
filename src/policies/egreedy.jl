using Random

#TODO for some reason Sarsa did not work with this policy. Used a q function and mual eps greed choice
mutable struct EpsilonGreedy{TP, TB} <: AbstractPolicy where {TP<:Real,TB<:AbstractFuncApprox}
    f::LinearFunction{TP, TB}
    action::Int
    vals::Array{TP,1}
    ϵ::TP
    function EpsilonGreedy(state_dim::Int, num_actions::Int, ϵ::T) where {T<:Real}
        f = LinearFunction(state_dim, num_actions)
        new{eltype(f.θ),typeof(f.ϕ)}(f, -1, zeros(eltype(f.θ), num_actions))
    end
    function EpsilonGreedy(::Type{T}, state_dim::Int, num_actions::Int, ϵ::T) where {T<:Real}
        f = LinearFunction(T, state_dim, num_actions)
        new{T,typeof(f.ϕ)}(f, -1, zeros(T, num_actions), ϵ)
    end
    function EpsilonGreedy(f::LinearFunction{T}, ϵ::T) where {T<:Real}
        new{T,typeof(f.ϕ)}(clone(f), -1, zeros(T, get_num_outputs(f)), ϵ)
    end
end

function get_num_params(π::EpsilonGreedy)::Int
    return get_num_params(π.f)
end

function get_action!(π::EpsilonGreedy{T}, x, rng::AbstractRNG)::T where {T<:Real}
    call!(π.vals, π.f, x)
    num_actions = length(π.vals)
    logp = π.ϵ * (1. / T(num_actions))
    if rand(rng, 1)[1] < π.ϵ
        π.action = argmax(π.vals)
        logp += (1. - π.ϵ)
    else
        π.action = rand(rng, 1:length(π.vals))[1]
    end
    return logp
end

function gradient_logp!(grad::Array{T}, π::EpsilonGreedy{T}, x, action::Int)::T where {T<:Real}
    grad = reshape(grad, size(get_params(π.f)))
    gradient!(grad, π.f, x, action)
    logp = 0.
    return logp
end


function get_action_gradient_logp!(grad::Array{T}, π::EpsilonGreedy{T}, x, rng::AbstractRNG)::T where {T<:Real}
    call_gradient!(π.vals, grad, π.f, x)
    num_actions = length(π.vals)
    logp = π.ϵ * (1. / T(num_actions))
    if rand(rng, 1)[1] < π.ϵ
        π.action = argmax(π.vals)
        logp += (1. - π.ϵ)
    else
        π.action = rand(rng, 1:length(π.vals))[1]
    end

    fill!(view(grad, :, 1:num_actions .!= π.action), 0.)
    return logp
end

function get_action_gradient_logp(grad::Array{T}, π::EpsilonGreedy{T}, x, rng::AbstractRNG) where {T<:Real}
    grad = zeros(T, size(get_params(π)))
    logp = get_action_gradient_logp!(grad, π, x)
    action = π.action
    return action, grad, logp
end

function set_params!(π::EpsilonGreedy{T}, θ::Array{T}) where {T<:Real}
    set_params!(π.f, θ)
end

function get_params(π::EpsilonGreedy)
    get_params(π.f)
end

function copy_params!(params::Array{T}, π::EpsilonGreedy{T}) where {T<:Real}
    copy_params!(params, π.f)
end

function copy_params(π::EpsilonGreedy{T})::Array{T} where {T<:Real}
    return copy_params(π.f)
end

function add_to_params!(π::EpsilonGreedy{T}, Δθ::Array{T}) where {T<:Real}
    add_to_params!(π.f, Δθ)
end

function clone(π::EpsilonGreedy)::EpsilonGreedy
    π₂ = EpsilonGreedy(π.f, π.ϵ)
    π₂.action = deepcopy(π.action)
    π₂.vals = deepcopy(π.vals)
    return π₂
end
