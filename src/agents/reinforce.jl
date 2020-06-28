
mutable struct REINFORCE{T,TP,TV} <: AbstractPolicyAgent where{T<:Real,TP<:AbstractPolicy,TV<:AbstractFuncApprox}
    π::TP
    vf::TV
    αₚ::T
    αᵥ::T
    γ::T
    avgGrad::Bool
    discountGrad::Bool
    states::Vector
    actions::Vector
    rewards::Array{T, 1}
    function REINFORCE(π::AbstractPolicy, vf::AbstractFuncApprox, αₚ::T, αᵥ::T, γ::T, avgGrad::Bool=false, discountGrad::Bool=true) where {T<:Real}
        new{T,typeof(π),typeof(vf)}(clone(π), clone(vf), αₚ, αᵥ, γ, avgGrad, discountGrad, [], [], T[])
    end
end

function act!(agent::REINFORCE, env::AbstractEnvironment, rng::AbstractRNG)
    push!(agent.states, env.state)

    get_action!(agent.π, env.state, rng)
    push!(agent.actions, agent.π.action)

    reward = step!(env, agent.π.action, rng)
    push!(agent.rewards, reward)

    if is_terminal(env)
        Gs = compute_returns(agent.rewards, agent.γ)
        update_withBaseline!(agent, Gs)
        new_episode!(agent, rng)
    end
end

function compute_returns(rewards::Array{T, 1}, γ::T)::Array{T, 1} where {T<:Real}
    G = 0.
    Gs = zeros(T, length(rewards))
    for t in length(rewards):-1:1
        G = rewards[t] + γ * G
        Gs[t] = G
    end
    return Gs
end


function update_withBaseline!(agent::REINFORCE, Gs::Array{T, 1}) where {T<:Real}
    γₜ = 1.0
    gₚ = zeros(get_num_params(agent.π))
    gᵥ = zeros(get_num_params(agent.vf))
    ∇ = zeros(get_num_params(agent.π))
    for t in 1:length(agent.states)
        v = call_gradient!(gᵥ, agent.vf, agent.states[t], 1)
        δ = Gs[t] - v
        @. gᵥ *= δ * agent.αᵥ
        add_to_params!(agent.vf, gᵥ)

        gradient_logp!(gₚ, agent.π, agent.states[t], agent.actions[t])
        @. ∇ += δ * agent.αₚ * γₜ * gₚ

        if agent.discountGrad
            γₜ *= agent.γ
        end
    end
    if agent.avgGrad
        ∇ ./= length(agent.states)
    end
    add_to_params!(agent.π, ∇)
end

function new_episode!(agent::REINFORCE, rng::AbstractRNG)
    empty!(agent.states)
    empty!(agent.actions)
    empty!(agent.rewards)
end

function clone(agent::REINFORCE)::REINFORCE
    a = REINFORCE(agent.π, agent.vf, agent.αₚ, agent.αᵥ, agent.γ, agent.avgGrad, agent.discountGrad)
    a.states = deepcopy(agent.states)
    a.actions = deepcopy(agent.actions)
    a.rewards = deepcopy(agent.rewards)
    return a
end
