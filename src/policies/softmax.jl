using LinearAlgebra
using Distributions, Random


#softmax(x) = exp.(x) / sum(exp.(x))
function softmax(x)
    x = clamp.(x, -32., 32.)
    return exp.(x) / sum(exp.(x))
end

function sample_discrete(p, rng)
    n = length(p)
    i = 1
    c = p[1]
    u = rand(rng)
    while c < u && i < n
        c += p[i += 1]
    end

    return i
end

mutable struct LinearSoftmaxPolicy{TP, TB} <: AbstractPolicy where {TP<:Real,TB<:AbstractFuncApprox}
    f::LinearFunction{TP, TB}
    action::Int
    probs::Array{TP,1}
    function LinearSoftmaxPolicy(state_dim::Int, num_actions::Int)
        f = LinearFunction(state_dim, num_actions)
        new{eltype(f.θ),typeof(f.ϕ)}(f, 0, zeros(eltype(f.θ), num_actions))
    end
    function LinearSoftmaxPolicy(::Type{T}, state_dim::Int, num_actions::Int) where {T<:Real}
        f = LinearFunction(T, state_dim, num_actions)
        new{T,typeof(f.ϕ)}(f, 0, zeros(T, num_actions))
    end
    function LinearSoftmaxPolicy(f::LinearFunction{T}) where {T<:Real}
        new{T,typeof(f.ϕ)}(clone(f), 0, zeros(T, get_num_outputs(f)))
    end
end

function get_num_params(π::LinearSoftmaxPolicy)::Int
    return get_num_params(π.f)
end

function get_action_probabilities!(π::LinearSoftmaxPolicy, x)
    call!(π.probs,π.f,x)
    π.probs .= softmax(π.probs)
end

function get_action_probabilities(π::LinearSoftmaxPolicy{T}, x)::Array{T,1} where {T<:Real}
    get_action_probabilities!(π, x)
    return π.probs
end

function get_action!(π::LinearSoftmaxPolicy{T}, x, rng::AbstractRNG)::T where {T<:Real}
    get_action_probabilities!(π, x)
    π.action = sample_discrete(π.probs, rng)
    logp = log(π.probs[π.action])
    return logp
end

function gradient_logp!(grad::Array{T}, π::LinearSoftmaxPolicy{T, IdentityBasis}, x::Int, action::Int)::T where {T<:Real}
    grad = reshape(grad, size(get_params(π.f)))
    fill!(grad, 0.)

    get_action_probabilities!(π, x)
    # probs = -π.probs
    logp = log(π.probs[action])
    # probs[action] += 1

    grad[x, :] .-= π.probs
    grad[x, action] += 1.

    return logp
end

function gradient_logp!(grad::Array{T}, π::LinearSoftmaxPolicy{T, IdentityBasis}, x::Array{T, 1}, action::Int)::T where {T<:Real}
    grad = reshape(grad, size(get_params(π.f)))
    fill!(grad, 0.)

    get_action_probabilities!(π, x)
    probs = -π.probs
    logp = log(π.probs[action])
    probs[action] += 1.

    for i in 1:length(probs)
        @. grad[:, i] = x * probs[i]
    end
    # grad .= x*probs'

    return logp
end

function gradient_logp!(grad::Array{T}, π::LinearSoftmaxPolicy{T}, x, action::Int)::T where {T<:Real}
    grad = reshape(grad, size(get_params(π.f)))
    fill!(grad, 0.)

    get_action_probabilities!(π, x)
    probs = -π.probs
    logp = log(π.probs[action])
    probs[action] += 1.
    for i in 1:length(probs)
        @. grad[:, i] = π.f.feats * probs[i]
    end
    # grad .= π.f.feats*probs'

    return logp
end



function gradient_entropy!(grad, π::LinearSoftmaxPolicy{T, IdentityBasis}, x::Int; reuse::Bool=false) where {T}
    grad = reshape(grad, size(get_params(π.f)))
    fill!(grad, 0.)
    if !reuse
        get_action_probabilities!(π, x)
    end

    p1 = zeros(length(π.probs))
    @. p1 = -π.probs * log(π.probs)
    H = sum(p1)  # entropy (p1 has negative in it)
    @. p1 += π.probs * H
    grad[x, :] .= p1
    return H
end

function gradient_entropy!(grad, π::LinearSoftmaxPolicy{T, IdentityBasis}, x::Array{T, 1}; reuse::Bool=false) where {T}
    grad = reshape(grad, size(get_params(π.f)))
    fill!(grad, 0.)
    if !reuse
        get_action_probabilities!(π, x)
    end

    p1 = zeros(length(π.probs))
    @. p1 = -π.probs * log(π.probs)
    H = sum(p1)  # entropy (p1 has negative in it)
    @. p1 += π.probs * H

    for i in 1:length(p1)
        @. grad[:, i] = x * p1[i]
    end
    # grad .= x*p1'
    return H
end

function gradient_entropy!(grad, π::LinearSoftmaxPolicy{T}, x; reuse::Bool=false) where {T}
    grad = reshape(grad, size(get_params(π.f)))
    fill!(grad, 0.)
    if !reuse
        get_action_probabilities!(π, x)
    end

    p1 = zeros(length(π.probs))
    @. p1 = -π.probs * log(π.probs)
    H = sum(p1)  # entropy (p1 has negative in it)
    @. p1 += π.probs * H

    for i in 1:length(p1)
        @. grad[:, i] = π.f.feats * p1[i]
    end

    # grad .= π.f.feats*p1'
    return H
end

function get_action_gradient_logp!(grad::Array{T}, π::LinearSoftmaxPolicy{T, IdentityBasis}, x::Int, rng::AbstractRNG)::T where {T<:Real}
    get_action_probabilities!(π, x)
    π.action = sample_discrete(π.probs, rng)
    logp = log(π.probs[π.action])
    # probs = -π.probs
    # probs[π.action] += 1.
    fill!(grad, 0.)
    grad[x, :] .-= π.probs
    grad[x, π.action] += 1.
    # grad[x, :] .= probs
    return logp
end

function get_action_gradient_logp!(grad::Array{T}, π::LinearSoftmaxPolicy{T, IdentityBasis}, x::Array{T, 1}, rng::AbstractRNG)::T where {T<:Real}
    get_action_probabilities!(π, x)
    π.action = sample_discrete(π.probs, rng)
    logp = log(π.probs[π.action])
    probs = -π.probs
    probs[π.action] += 1.
    grad .= x*probs'
    return logp
end

function get_action_gradient_logp!(grad::Array{T}, π::LinearSoftmaxPolicy{T}, x, rng::AbstractRNG)::T where {T<:Real}
    get_action_probabilities!(π, x)
    π.action = sample_discrete(π.probs, rng)
    logp = log(π.probs[π.action])
    probs = deepcopy(-π.probs)
    probs[π.action] += 1.
    grad .= π.f.feats*probs'
    return logp
end

function get_action_gradient_logp(grad::Array{T}, π::LinearSoftmaxPolicy{T}, x, rng::AbstractRNG) where {T<:Real}
    grad = zeros(T, size(get_params(π)))
    logp = get_action_gradient_logp!(grad, π, x)
    action = π.action
    return action, grad, logp
end

function set_params!(π::LinearSoftmaxPolicy{T}, θ) where {T<:Real}
    set_params!(π.f, θ)
end

function get_params(π::LinearSoftmaxPolicy)
    get_params(π.f)
end

function copy_params!(params::Array{T}, π::LinearSoftmaxPolicy{T}) where {T<:Real}
    copy_params!(params, π.f)
end

function copy_params(π::LinearSoftmaxPolicy{T})::Array{T} where {T<:Real}
    return copy_params(π.f)
end

function add_to_params!(π::LinearSoftmaxPolicy{T}, Δθ) where {T<:Real}
    add_to_params!(π.f, Δθ)
end

function clone(π::LinearSoftmaxPolicy)::LinearSoftmaxPolicy
    π₂ = LinearSoftmaxPolicy(π.f)
    π₂.action = deepcopy(π.action)
    π₂.probs = deepcopy(π.probs)
    return π₂
end
