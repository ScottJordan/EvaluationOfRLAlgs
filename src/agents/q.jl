using LinearAlgebra

mutable struct QLearning{T,TQ,OQ} <: AbstractAgent where {T<:Real,TQ<:AbstractPolicy,OQ<:AbstractOptimizer}
    qf::TQ
    qopt::OQ
    γ::T
    ϵ::T
    gp::Array{T, 2}
    action::Int

    function QLearning(qf::AbstractFuncApprox, qopt::AbstractOptimizer, γ::T, ϵ::T) where {T<:Real}
        tp = eltype(get_params(qf))
        sp = size(get_params(qf))
        qf2 = clone(qf)
        qopt2 = clone(qopt, qf2)
        new{tp,typeof(qf2),typeof(qopt2)}(qf2, qopt2, tp(γ), tp(ϵ), zeros(tp, sp), -1)
    end
end

function act!(agent::TA, env::TE, rng::AbstractRNG, learn::Bool) where {TA<:QLearning,TE<:AbstractEnvironment} # do not learn if learn flag is set
    qs = zeros(get_num_outputs(agent.qf))
    call!(qs, agent.qf, env.state)

    if rand(rng, 1)[1] < agent.ϵ
        agent.action = rand(rng, 1:length(qs))[1]
    else
        agent.action = argmax(qs)
    end

    r = step!(env, agent.action, rng)
end

function act!(agent::QLearning, env::AbstractEnvironment, rng::AbstractRNG)
    x = env.state
    fill!(agent.gp, 0.)
    q = 0.
    if agent.action != -1
        q = call_gradient!(agent.gp, agent.qf, env.state, agent.action)
    else
        qs = zeros(get_num_outputs(agent.qf))
        call_gradient!(qs, agent.gp, agent.qf, env.state)

        if rand(rng, 1)[1] < agent.ϵ
            agent.action = rand(rng, 1:length(qs))[1]
        else
            agent.action = argmax(qs)
        end
        q = qs[agent.action]
        agent.gp[:, 1:end .!= agent.action] .= 0.
    end

    r = step!(env, agent.action, rng)

    if is_terminal(env)
        target = r
    else
        qs = call(agent.qf, env.state)
        probs = softmax(qs)

        if rand(rng, 1)[1] < agent.ϵ
            agent.action = rand(rng, 1:length(qs))[1]
        else
            agent.action = argmax(qs)
        end

        #TD update
        q2 = maximum(qs)
        target = r + agent.γ * q2
    end

    delta = target - q

    update!(agent.qopt, delta, agent.gp)
    if is_terminal(env)
        new_episode!(agent, rng)
    end
end

function act!(agent::QLearning{T,TQ,TO}, env::AbstractEnvironment, rng::AbstractRNG) where {T,TQ,TO<:Union{Parl2AccumulatingTraceOptimizer{T,TQ},AutoTIDBDOptimizer}}
    x = env.state
    fill!(agent.gp, 0.)
    q = 0.
    if agent.action != -1
        q = call_gradient!(agent.gp, agent.qf, env.state, agent.action)
    else
        qs = zeros(get_num_outputs(agent.qf))
        call_gradient!(qs, agent.gp, agent.qf, env.state)

        if rand(rng, 1)[1] < agent.ϵ
            agent.action = rand(rng, 1:length(qs))[1]
        else
            agent.action = argmax(qs)
        end
        q = qs[agent.action]
        agent.gp[:, 1:end .!= agent.action] .= 0.
    end

    r = step!(env, agent.action, rng)
    gp2 = zeros(T, size(agent.gp))
    if is_terminal(env)
        target = r
    else
        qs = call(agent.qf, env.state)
        probs = softmax(qs)

        if rand(rng, 1)[1] < agent.ϵ
            agent.action = rand(rng, 1:length(qs))[1]
        else
            agent.action = argmax(qs)
        end
        gradient!(gp2, agent.qf, env.state, agent.action)
        #TD update
        q2 = maximum(qs)
        target = r + agent.γ * q2
    end

    delta = target - q

    update!(agent.qopt, delta, agent.gp, gp2)
    if is_terminal(env)
        new_episode!(agent, rng)
    end
end

function new_episode!(agent::QLearning, rng::AbstractRNG)
    new_episode!(agent.qopt)
    agent.action = -1
end

function clone(agent::QLearning)::QLearning
    a = QLearning(agent.qf, agent.qopt, agent.γ, agent.ϵ)
    return a
end
