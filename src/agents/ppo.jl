import Base.GC.gc
import Base.summarysize

mutable struct PPO{T,TP,TV,OP,OV} <: AbstractPolicyAgent where{T<:Real,TP<:AbstractPolicy,TV<:AbstractFuncApprox,OP<:AbstractOptimizer,OV<:AbstractOptimizer}
    π::TP
    vf::TV
    popt::OP
    vopt::OV
    γ::T
    λ::T
    ϵ::T  # clip param
    α::T  # entropy
    steps_per_batch::Int
    num_epochs::Int
    batch_size::Int
    counter::Int
    states::Vector
    actions::Vector
    blogps::Array{T, 1}
    rewards::Array{T, 1}
    values::Array{T, 1}
    terminals::Array{Bool,1}
    gp::Array{T}
    ψ::Array{T}
    gv::Array{T}
    ϕ::Array{T}

    function PPO(π::AbstractPolicy, vf::AbstractFuncApprox, popt::AbstractOptimizer, vopt::AbstractOptimizer, γ::T, λ::T, ϵ::T, α::T, steps_per_batch::Int, num_epochs::Int, batch_size::Int, ::Type{TS}, ::Type{TA}) where {T<:Real,TS,TA}
        π2 = clone(π)
        vf2 = clone(vf)
        popt2 = clone(popt, π2)
        vopt2 = clone(vopt, vf2)
        states = Array{TS, 1}(undef, steps_per_batch)
        actions = Array{TA, 1}(undef, steps_per_batch)
        blogps = Array{T, 1}(undef, steps_per_batch)
        rewards = Array{T, 1}(undef, steps_per_batch)
        values = Array{T, 1}(undef, steps_per_batch+1)
        terminals = Array{Bool, 1}(undef, steps_per_batch)
        gp = zeros(size(get_params(π)))
        ψ = zeros(size(get_params(π)))
        gv = zeros(size(get_params(vf)))
        ϕ = zeros(size(get_params(vf)))
        new{T,typeof(π),typeof(vf),typeof(popt),typeof(vopt)}(π2, vf2, popt2, vopt2, γ, λ, ϵ, α, steps_per_batch, num_epochs, batch_size, 0, states, actions, blogps, rewards, values, terminals, gp, ψ, gv, ϕ)
    end
end

function act!(agent::PPO, env::AbstractEnvironment, rng::AbstractRNG)
    agent.counter += 1
    counter = agent.counter
    # push!(agent.states, env.state)
    agent.states[counter] = deepcopy(env.state)
    logp = get_action!(agent.π, env.state, rng)
    # push!(agent.actions, agent.π.action)
    # push!(agent.blogps, logp)
    agent.actions[counter] = deepcopy(agent.π.action)
    agent.blogps[counter] = logp

    v = call(agent.vf, env.state, 1)
    # push!(agent.values, v)
    agent.values[counter] = v

    reward = step!(env, agent.π.action, rng)
    # push!(agent.rewards, reward)
    agent.rewards[counter] = reward
    terminal = is_terminal(env)
    # push!(agent.terminals, terminal)
    agent.terminals[counter] = terminal
    if counter ≥ agent.steps_per_batch #length(agent.states) ≥ agent.steps_per_batch
        if !terminal
            v′ = call(agent.vf, env.state, 1)
            # push!(agent.values, v′)
            agent.values[counter+1] = v′
        else
            # push!(agent.values, 0.0)
            agent.values[counter+1] = 0.0
        end

        ppo_update!(agent, rng)

        # empty!(agent.states)
        # empty!(agent.actions)
        # empty!(agent.blogps)
        # empty!(agent.rewards)
        # empty!(agent.values)
        # empty!(agent.terminals)
        agent.counter = 0
    end
end

function ppo_make_targets!(LamAdvs, LamRets, γ, λ, rewards, values, terminals)
    prevAdv = 0.0
    T = length(rewards)
    for t in T:-1:1
        r = rewards[t]
        v = values[t]
        v′ =values[t+1]
        terminal = terminals[t]
        δ = r + γ * v′ * (1-terminal) - v
        LamAdvs[t] = δ + γ * λ * (1-terminal) * prevAdv
        LamRets[t] = LamAdvs[t] + v
        prevAdv = LamAdvs[t]
    end
end

function ppo_policy_grad!(g, ψ, π, state, action, blogp, Adv, ϵ, α)
    logp = gradient_logp!(ψ, π, state, action)
    ratio = exp(logp - blogp)
    unclipped = ratio * Adv
    clipped = clamp(ratio, 1.0 - ϵ, 1.0 + ϵ) * Adv
    if (unclipped ≤ clipped)  # if clipped is lower then the ratio has been clipped and gradient would be 0
        @. g += unclipped * ψ
    end
    if α > 0
        gradient_entropy!(ψ, π, state, reuse=true)
        @. g += α * ψ
    end
end

function ppo_value_grad!(g, ϕ, vf, state, G)
    v = call_gradient!(ϕ, vf, state, 1)
    δ = G - v
    @. g += δ * ϕ
end



function ppo_update!(agent::PPO, rng)
    N = length(agent.blogps)
    LamAdvs = zeros(N)
    LamRets = zeros(N)

    ppo_make_targets!(LamAdvs, LamRets, agent.γ, agent.λ, agent.rewards, agent.values, agent.terminals)

    # gp = zeros(size(get_params(agent.π)))
    # ψ = zeros(size(get_params(agent.π)))
    # gv = zeros(size(get_params(agent.vf)))
    # ϕ = zeros(size(get_params(agent.vf)))
    # gp = agent.gp
    # ψ = agent.ψ
    # gv = agent.gv
    # ϕ = agent.ϕ
    for epoch in 1:agent.num_epochs
        idxs = randperm(rng, N)
        i = 1
        count = 0
        while i <= N
            t = idxs[i]
            ppo_policy_grad!(agent.gp, agent.ψ, agent.π, agent.states[t], agent.actions[t], agent.blogps[t], LamAdvs[t], agent.ϵ, agent.α)
            ppo_value_grad!(agent.gv, agent.ϕ, agent.vf, agent.states[t], LamRets[t])
            count += 1
            if count >= agent.batch_size || i == N
                @. agent.gp /= count
                @. agent.gv /= count
                update!(agent.popt, agent.gp)
                update!(agent.vopt, agent.gv)
                count = 0
                fill!(agent.gp, 0.0)
                fill!(agent.gv, 0.0)
            end
            i += 1
        end
    end
    # gc(true)
    # gc(true)
end


function new_episode!(agent::PPO, rng::AbstractRNG)
    nothing
end

function clone(agent::PPO)::PPO
    a = PPO(agent.π, agent.vf, agent.popt, agent.vopt, agent.γ, agent.λ, agent.ϵ, agent.α, agent.steps_per_batch, agent.num_epochs, agent.batch_size, eltype(agent.states), eltype(agent.actions))
    a.states = deepcopy(agent.states)
    a.actions = deepcopy(agent.actions)
    a.blogps = deepcopy(agent.blogps)
    a.rewards = deepcopy(agent.rewards)
    a.values = deepcopy(agent.values)
    a.terminals = deepcopy(agent.terminals)
    a.counter = agent.counter
    a.gp .= agent.gp
    a.gv .= agent.gv
    a.ψ .= agent.ψ
    a.ϕ .= agent.ϕ
    return a
end
