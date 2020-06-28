using LinearAlgebra

mutable struct ActorCritic{T,TP,TV,OP,OV} <: AbstractPolicyAgent where {T<:Real,TP<:AbstractPolicy,TV<:AbstractFuncApprox,OP<:AbstractOptimizer,OV<:AbstractOptimizer}
    π::TP
    vf::TV
    popt::OP
    vopt::OV
    γ::T
    gp::Array{T}
    gv::Array{T}
    function ActorCritic(π::AbstractPolicy, vf::AbstractFuncApprox, popt::AbstractOptimizer, vopt::AbstractOptimizer, γ::T) where {T<:Real}
        tp = eltype(get_params(π))
        sp = size(get_params(π))
        sv = size(get_params(vf))
        π2 = clone(π)
        vf2 = clone(vf)
        popt2 = clone(popt, π2)
        vopt2 = clone(vopt, vf2)
        new{tp,typeof(π2),typeof(vf2),typeof(popt2),typeof(vopt2)}(π2, vf2, popt2, vopt2, tp(γ), zeros(tp, sp), zeros(tp, sv))
    end
end

function act!(agent::ActorCritic, env::AbstractEnvironment, rng::AbstractRNG)
    v = call_gradient!(agent.gv, agent.vf, env.state, 1)
    logp = get_action_gradient_logp!(agent.gp, agent.π, env.state, rng)

    r = step!(env, agent.π.action, rng)
    target = 0.
    vnext = 0.
    if is_terminal(env)
        target = r
    else
        vnext = call(agent.vf, env.state, 1)
        target = r + agent.γ * vnext
    end

    delta = target - v

    update!(agent.vopt, delta, agent.gv)
    update!(agent.popt, delta, agent.gp)

    if is_terminal(env)
        new_episode!(agent, rng)
    end
end

function act!(agent::ActorCritic{T,TP,TV,OP,Parl2AccumulatingTraceOptimizer{T,TV}}, env::AbstractEnvironment, rng::AbstractRNG) where {T,TP,TV,OP}
    v = call_gradient!(agent.gv, agent.vf, env.state, 1)
    logp = get_action_gradient_logp!(agent.gp, agent.π, env.state, rng)

    r = step!(env, agent.π.action, rng)
    target = 0.
    vnext = 0.
    gvnext = zeros(T, size(agent.gv))
    if is_terminal(env)
        target = r
    else
        vnext = call_gradient!(gvnext, agent.vf, env.state, 1)
        target = r + agent.γ * vnext
    end

    delta = target - v - 0.01 * logp

    update!(agent.vopt, delta, agent.gv, gvnext)
    update!(agent.popt, delta, agent.gp)

    if is_terminal(env)
        new_episode!(agent, rng)
    end
end


function new_episode!(agent::ActorCritic, rng::AbstractRNG)
    new_episode!(agent.vopt)
    new_episode!(agent.popt)
end

function clone(agent::ActorCritic)::ActorCritic
    a = ActorCritic(agent.π, agent.vf, agent.popt, agent.vopt, agent.γ)
    return a
end
