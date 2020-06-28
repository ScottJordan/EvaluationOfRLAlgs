using LinearAlgebra

mutable struct NACTD{T,TP,TV,TW,OP,OV,OW} <: AbstractPolicyAgent where {T<:Real,TP<:AbstractPolicy,TV<:AbstractFuncApprox,TW<:AbstractFuncApprox,OP<:AbstractOptimizer,OV<:AbstractOptimizer,OW<:AbstractOptimizer}
    π::TP
    vf::TV
    wf::TW
    popt::OP
    vopt::OV
    wopt::OW
    γ::T
    normalizew::Bool
    gp::Array{T, 2}
    gv::Array{T, 2}
    function NACTD(π::AbstractPolicy, vf::AbstractFuncApprox, wf::AbstractFuncApprox, popt::AbstractOptimizer, vopt::AbstractOptimizer, wopt::AbstractOptimizer, γ::T, normalizew::Bool) where {T<:Real}
        tp = eltype(get_params(π))
        sp = size(get_params(π))
        sv = size(get_params(vf))

        π2 = clone(π)
        vf2 = clone(vf)
        wf2 = clone(wf)
        popt2 = clone(popt, π2)
        vopt2 = clone(vopt, vf2)
        wopt2 = clone(wopt, wf2)
        new{tp,typeof(π2),typeof(vf2),typeof(wf2),typeof(popt2),typeof(vopt2),typeof(wopt2)}(π2, vf2, wf2, popt2, vopt2, wopt2, tp(γ), normalizew, zeros(tp, sp), zeros(tp, sv))
    end
end

function act!(agent::NACTD, env::AbstractEnvironment, rng::AbstractRNG)
    v = call_gradient!(agent.gv, agent.vf, env.state, 1)
    logp = get_action_gradient_logp!(agent.gp, agent.π, env.state, rng)
    adv = call(agent.wf, vec(agent.gp), 1)
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
    delta_adv = delta - adv
    update!(agent.vopt, delta, agent.gv)
    update!(agent.wopt, delta_adv, vec(agent.gp))

    w = vec(get_params(agent.wf))

    if agent.normalizew
        w = w .* (1. / norm(vec(w)))
    end

    update!(agent.popt, w)

    if is_terminal(env)
        new_episode!(agent, rng)
    end
end

function act!(agent::NACTD{T,TP,TV,TW,OP,Parl2AccumulatingTraceOptimizer{T,TV}}, env::AbstractEnvironment, rng::AbstractRNG) where {T,TP,TV,TW,OP}
    v = call_gradient!(agent.gv, agent.vf, env.state, 1)
    logp = get_action_gradient_logp!(agent.gp, agent.π, env.state, rng)
    adv = call(agent.wf, vec(agent.gp), 1)
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

    delta = target - v
    delta_adv = delta - adv

    update!(agent.vopt, delta, agent.gv, gvnext)
    update!(agent.wopt, delta_adv, vec(agent.gp))

    w = vec(get_params(agent.wf))

    if agent.normalizew
        w = w .* (1. / norm(vec(w)))
    end

    if is_terminal(env)
        new_episode!(agent, rng)
    end
end


function new_episode!(agent::NACTD, rng::AbstractRNG)
    new_episode!(agent.vopt)
    new_episode!(agent.wopt)
    new_episode!(agent.popt)
end

function clone(agent::NACTD)::NACTD
    a = NACTD(agent.π, agent.vf, agent.wf, agent.popt, agent.vopt, agent.wopt, agent.γ, agent.normalizew)
    return a
end
