using Statistics

abstract type AbstractFiniteDifferencePolicySearch <: AbstractPolicyAgent end

# finite differences policy gradient search
mutable struct BasicRandomSearch{T,TP} <: AbstractFiniteDifferencePolicySearch where{T<:Real,TP<:AbstractPolicy}
    π::TP
    α::T
    γ::T
    ν::T
    num_samples::Int
    max_episodes
    θ::Array{T}
    δs::Array{T}
    Gs::Array{T}
    ep_number::Int
    dnum::Int
    pnum::Int
    G::T
    cur_gamma::T

    function BasicRandomSearch(π::AbstractPolicy, α::T, γ::T, ν::T, num_samples::Int, rng::AbstractRNG, max_episodes=Inf) where {T<:Real}
        θ = deepcopy(get_params(π))
        δs = ν .* randn(rng, (num_samples, size(θ)...))
        Gs = zeros((num_samples, 2))
        dnum = 1
        ep_number = 1
        pnum = 1
        new{T,typeof(π)}(clone(π), α, γ, ν, num_samples, max_episodes, θ, δs, Gs, ep_number, dnum, pnum, 0., 1.)
    end

    function BasicRandomSearch(π::AbstractPolicy, α::T, γ::T, ν::T, num_samples::Int, max_episodes=Inf) where {T<:Real}
        θ = deepcopy(get_params(π))
        δs = zeros(T, (num_samples, size(θ)...))
        Gs = zeros(T, (num_samples, 2))
        dnum = 1
        ep_number = 1
        pnum = 1
        new{T,typeof(π)}(clone(π), α, γ, ν, num_samples, max_episodes, θ, δs, Gs, ep_number, dnum, pnum, 0., 1.)
    end
end

function act!(agent::AbstractFiniteDifferencePolicySearch, env::AbstractEnvironment, rng::AbstractRNG)
    get_action!(agent.π, env.state, rng)

    reward = step!(env, agent.π.action, rng)

    agent.G += agent.cur_gamma * reward
    agent.cur_gamma *= agent.γ

    if is_terminal(env)
        agent.Gs[agent.dnum, agent.pnum] = agent.G
        agent.G = 0
        agent.ep_number += 1
        if agent.pnum == 1
            agent.pnum = 2
        else
            agent.pnum = 1
            agent.dnum += 1
        end
    end
end


function new_episode!(agent::AbstractFiniteDifferencePolicySearch, rng::AbstractRNG)
    agent.G = 0.
    agent.cur_gamma = 1.0

    set_params!(agent.π, agent.θ)

    if agent.ep_number ≤ agent.max_episodes
        if agent.dnum > agent.num_samples
            update_theta!(agent)
            agent.δs .= agent.ν .* randn(rng, (agent.num_samples, size(agent.θ)...))
            agent.dnum = 1
            agent.pnum = 1
        end

        Δ = view(reshape(agent.δs, (agent.num_samples, :)), agent.dnum, :)
        if agent.pnum == 1
            add_to_params!(agent.π, -Δ)
        else
            add_to_params!(agent.π, Δ)
        end
    end

end

function update_theta!(agent::BasicRandomSearch)
    θ = agent.θ
    Gs = agent.Gs
    δs = agent.δs

    g = (view(Gs, :, 2) .- view(Gs, :, 1)) .* reshape(δs, (agent.num_samples, :))
    ḡ =  reshape(mean(g, dims=1), size(θ))
    @. agent.θ += agent.α * ḡ
end

function clone(agent::BasicRandomSearch)::BasicRandomSearch
    a = BasicRandomSearch(agent.π, agent.α, agent.γ, agent.ν, agent.num_samples, agent.max_episodes)
    a.θ .= agent.θ
    a.δs .= agent.δs
    a.Gs .= agent.Gs
    a.ep_number = agent.ep_number
    a.dnum = agent.dnum
    a.pnum = agent.pnum
    a.G = agent.G
    a.cur_gamma = agent.cur_gamma

    return a
end


mutable struct AugmentedRandomSearch{T,TP} <: AbstractFiniteDifferencePolicySearch where{T<:Real,TP<:AbstractPolicy}
    π::TP
    α::T
    γ::T
    ν::T
    num_samples::Int
    num_elite::Int
    max_episodes
    θ::Array{T}
    δs::Array{T}
    Gs::Array{T}
    ep_number::Int
    dnum::Int
    pnum::Int
    G::T
    cur_gamma::T

    function AugmentedRandomSearch(π::AbstractPolicy, α::T, γ::T, ν::T, num_samples::Int, num_elite::Int, rng::AbstractRNG, max_episodes=Inf) where {T<:Real}
        θ = deepcopy(get_params(π))
        δs = ν .* randn(rng, (num_samples, size(θ)...))
        Gs = zeros((num_samples, 2))
        dnum = 1
        ep_number = 1
        pnum = 1
        new{T,typeof(π)}(clone(π), α, γ, ν, num_samples, num_elite, max_episodes, θ, δs, Gs, ep_number, dnum, pnum, 0., 1.)
    end

    function AugmentedRandomSearch(π::AbstractPolicy, α::T, γ::T, ν::T, num_samples::Int, num_elite::Int, max_episodes=Inf) where {T<:Real}
        θ = deepcopy(get_params(π))
        δs = zeros(T, (num_samples, size(θ)...))
        Gs = zeros(T, (num_samples, 2))
        dnum = 1
        ep_number = 1
        pnum = 1
        new{T,typeof(π)}(clone(π), α, γ, ν, num_samples, num_elite, max_episodes, θ, δs, Gs, ep_number, dnum, pnum, 0., 1.)
    end
end

function update_theta!(agent::AugmentedRandomSearch)
    θ = agent.θ
    Gs = agent.Gs
    δs = agent.δs

    idxs = sortperm(vec(maximum(Gs, dims=2)), rev=true)[1:agent.num_elite]

    g  = (view(Gs, idxs, 2) .- view(Gs, idxs, 1)) .* view(reshape(δs, (agent.num_samples, :)), idxs, :)
    ḡ  = mean(g, dims=1)
    ḡ  = reshape(ḡ, size(θ))
    σᵣ = std(view(Gs, idxs, :))  # standard deviation of the 2*num_elite returns used
    @. agent.θ += agent.α / (σᵣ + 1e-8) * ḡ
end

function clone(agent::AugmentedRandomSearch)::AugmentedRandomSearch
    a = AugmentedRandomSearch(agent.π, agent.α, agent.γ, agent.ν, agent.num_samples, agent.num_elite, agent.max_episodes)
    a.θ .= agent.θ
    a.δs .= agent.δs
    a.Gs .= agent.Gs
    a.ep_number = agent.ep_number
    a.dnum = agent.dnum
    a.pnum = agent.pnum
    a.G = agent.G
    a.cur_gamma = agent.cur_gamma

    return a
end
