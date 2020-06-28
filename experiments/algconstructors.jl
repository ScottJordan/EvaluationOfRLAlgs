using Distributions
import EvaluationOfRLAlgs

function sample_fourierbasis(dorder_range, iorder_range, state_ranges, rng)
    sdim = size(state_ranges)[1]
    logp = 0.

    maxParams = 10000 # max params for dorder is 10k
    domp = max(floor((Float64(maxParams) + 1.0)^(1.0 / Float64(sdim)) - 1.), 0.)  # largest dorder that produces <= maxParams features
    maxDorder = min(dorder_range[2], domp)

    dorder = rand(rng, dorder_range[1]:dorder_range[2])
    logp += log(1. / (maxDorder - dorder_range[1] + 1))
    iorder = rand(rng, iorder_range[1]:iorder_range[2])
    logp += log(1. / (iorder_range[2] - iorder_range[1] + 1))

    # full = rand(rng, [true, false])
    # logp += log(0.5)
    full = false
    ϕ = EvaluationOfRLAlgs.FourierBasis(state_ranges, dorder, iorder, full)
    return ϕ, [dorder, iorder,full], logp
end

function logRand(low, high, rng)
    X = exp(rand(rng, Uniform(log(low), log(high))))
    logp = -log(log(high) - log(high) - log(X))
    return X, logp
end

function log2RandDisc(low, high, rng)
    X = round(Int,2^(rand(rng, Uniform(log2(low), log2(high)))))
    lxlow, lxhigh = log2(X), log2(X+1)
    Fxhigh = (lxhigh - log2(low)) / (log2(high) - log2(low))
    Fxlow = (lxlow - log2(low)) / (log2(high) - log2(low))
    logp = log(Fxhigh-Fxlow)
    return X, logp
end

function CreateSarsaLambda(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
    else
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)


    ϵrange = [0.0, 0.1]
    ϵ = rand(rng, Uniform(ϵrange[1], ϵrange[2]))
    logp += log(1.0 / (ϵrange[2] - ϵrange[1]))
    push!(hyps, ϵ)

    if discrete_states
        qarange = [1e-3, 0.1]
    else
        qarange = [1e-6, 0.001]
    end
    αq, qalogp = logRand(qarange[1], qarange[2], rng)
    logp += qalogp
    push!(hyps, αq)
    qopt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(qf, αq, γ, λ)

    agent = EvaluationOfRLAlgs.Sarsa(qf, qopt, γ, ϵ)
    return agent, hyps, logp
end

function CreateSarsaLambdaScaled(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
    else
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)


    ϵrange = [0.0, 0.1]
    ϵ = rand(rng, Uniform(ϵrange[1], ϵrange[2]))
    logp += log(1.0 / (ϵrange[2] - ϵrange[1]))
    push!(hyps, ϵ)

    if discrete_states
        qarange = [1e-3, 0.1]
    else
        phi_dim = EvaluationOfRLAlgs.get_num_outputs(ϕ)
        qarange = [1e-3, 1.0] ./ phi_dim
    end
    αq, qalogp = logRand(qarange[1], qarange[2], rng)
    logp += qalogp
    push!(hyps, αq)
    qopt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(qf, αq, γ, λ)

    agent = EvaluationOfRLAlgs.Sarsa(qf, qopt, γ, ϵ)
    return agent, hyps, logp
end


function CreateSarsaParl2(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
    else
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)


    ϵrange = [0.0, 0.1]
    ϵ = rand(rng, Uniform(ϵrange[1], ϵrange[2]))
    logp += log(1.0 / (ϵrange[2] - ϵrange[1]))
    push!(hyps, ϵ)

    qopt = EvaluationOfRLAlgs.Parl2AccumulatingTraceOptimizer(qf, γ, λ)

    agent = EvaluationOfRLAlgs.Sarsa(qf, qopt, γ, ϵ)
    return agent, hyps, logp
end

function CreateQLambda(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
    else
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)


    ϵrange = [0.0, 0.1]
    ϵ = rand(rng, Uniform(ϵrange[1], ϵrange[2]))
    logp += log(1.0 / (ϵrange[2] - ϵrange[1]))
    push!(hyps, ϵ)

    if discrete_states
        qarange = [1e-3, 0.1]
    else
        qarange = [1e-6, 0.001]
    end
    αq, qalogp = logRand(qarange[1], qarange[2], rng)
    logp += qalogp
    push!(hyps, αq)
    qopt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(qf, αq, γ, λ)

    agent = EvaluationOfRLAlgs.QLearning(qf, qopt, γ, ϵ)
    return agent, hyps, logp
end

function CreateQLambdaScaled(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
    else
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)


    ϵrange = [0.0, 0.1]
    ϵ = rand(rng, Uniform(ϵrange[1], ϵrange[2]))
    logp += log(1.0 / (ϵrange[2] - ϵrange[1]))
    push!(hyps, ϵ)

    if discrete_states
        qarange = [1e-3, 0.1]
    else
        phi_dim = EvaluationOfRLAlgs.get_num_outputs(ϕ)
        qarange = [1e-3, 1.0] ./ phi_dim
    end
    αq, qalogp = logRand(qarange[1], qarange[2], rng)
    logp += qalogp
    push!(hyps, αq)
    qopt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(qf, αq, γ, λ)

    agent = EvaluationOfRLAlgs.QLearning(qf, qopt, γ, ϵ)
    return agent, hyps, logp
end


function CreateQParl2(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
    else
        qf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)


    ϵrange = [0.0, 0.1]
    ϵ = rand(rng, Uniform(ϵrange[1], ϵrange[2]))
    logp += log(1.0 / (ϵrange[2] - ϵrange[1]))
    push!(hyps, ϵ)

    qopt = EvaluationOfRLAlgs.Parl2AccumulatingTraceOptimizer(qf, γ, λ)

    agent = EvaluationOfRLAlgs.QLearning(qf, qopt, γ, ϵ)
    return agent, hyps, logp
end

function CreateActorCritic(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    else
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)

    if discrete_states
        varange = [1e-3, 0.1]
        parange = [1e-3, 0.1]
    else
        varange = [1e-6, 1e-3]
        parange = [1e-6, 1e-3]
    end
    αp, palogp = logRand(parange[1], parange[2], rng)
    αv, valogp = logRand(varange[1], varange[2], rng)
    logp += palogp
    logp += valogp
    push!(hyps, αp)
    push!(hyps, αv)
    popt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(policy, αp, γ, λ)
    vopt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(vf, αv, γ, λ)

    agent = EvaluationOfRLAlgs.ActorCritic(policy, vf, popt, vopt, γ)
    return agent, hyps, logp
end

function CreateActorCriticScaled(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    else
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)

    if discrete_states
        varange = [1e-3, 0.1]
        parange = [1e-3, 0.1]
    else
        phi_dim = EvaluationOfRLAlgs.get_num_outputs(ϕ)
        varange = [1e-3, 1.0] ./ phi_dim
        parange = [1e-3, 1.0] ./ (phi_dim*num_actions)
    end
    αp, palogp = logRand(parange[1], parange[2], rng)
    αv, valogp = logRand(varange[1], varange[2], rng)
    logp += palogp
    logp += valogp
    push!(hyps, αp)
    push!(hyps, αv)
    popt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(policy, αp, γ, λ)
    vopt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(vf, αv, γ, λ)

    agent = EvaluationOfRLAlgs.ActorCritic(policy, vf, popt, vopt, γ)
    return agent, hyps, logp
end

function CreateActorCriticParl2(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    else
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)

    if discrete_states
        parange = [1e-3, 0.1]
    else
        phi_dim = EvaluationOfRLAlgs.get_num_outputs(ϕ)
        parange = [1e-3, 1.0] ./ (phi_dim*num_actions)
    end
    αp, palogp = logRand(parange[1], parange[2], rng)
    logp += palogp
    push!(hyps, αp)
    popt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(policy, αp, γ, λ)
    vopt =  EvaluationOfRLAlgs.Parl2AccumulatingTraceOptimizer(vf, γ, λ)

    agent = EvaluationOfRLAlgs.ActorCritic(policy, vf, popt, vopt, γ)
    return agent, hyps, logp
end

function CreateNACTD(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    else
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    end
    wf = EvaluationOfRLAlgs.LinearFunction(Float64, EvaluationOfRLAlgs.get_num_params(policy), 1)

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)

    if discrete_states
        varange = [1e-3, 0.1]
        warange = [1e-3, 0.1]
        parange = [1e-3, 0.1]
    else
        varange = [1e-6, 1e-3]
        warange = [1e-6, 1e-3]
        parange = [1e-6, 1e-3]
    end
    αp, palogp = logRand(parange[1], parange[2], rng)
    αv, valogp = logRand(varange[1], varange[2], rng)
    αw, walogp = logRand(warange[1], warange[2], rng)
    logp += palogp
    logp += valogp
    logp += walogp
    push!(hyps, αp)
    push!(hyps, αw)
    push!(hyps, αv)
    popt = EvaluationOfRLAlgs.SGA(policy, αp)
    wopt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(wf, αw, γ, λ)
    vopt = EvaluationOfRLAlgs.AccumulatingTraceOptimizer(vf, αv, γ, λ)

    # normalizew = rand(rng, [true, false])
    # logp += log(0.5)
    normalizew = true
    push!(hyps, normalizew)
    agent = EvaluationOfRLAlgs.NACTD(policy, vf, wf, popt, vopt, wopt, γ, Bool(normalizew))
    return agent, hyps, logp
end

function CreatePPO(env, rng)
    sdesc = EvaluationOfRLAlgs.get_state_desc(env)
    num_actions = EvaluationOfRLAlgs.get_action_desc(env)
    hyps = []
    logp = 0.0
    if typeof(sdesc) <: AbstractArray
        discrete_states = false
    else
        discrete_states = true
    end
    if !discrete_states
        dorder_range = [0,9]
        iorder_range = [1,9]
        ϕ, bhyps, blogp = sample_fourierbasis(dorder_range, iorder_range, sdesc, rng)
        append!(hyps, bhyps)
        logp += blogp
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    else
        vf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, 1)
        pf = EvaluationOfRLAlgs.LinearFunction(Float64, sdesc, num_actions)
        policy = EvaluationOfRLAlgs.LinearSoftmaxPolicy(pf)
    end

    # sample gamma
    Γ = EvaluationOfRLAlgs.get_gamma(env)
    γrange = (1e-4, 0.05)
    gamma, glogp = logRand(γrange[1], γrange[2], rng)
    logp += glogp
    γ = Γ - gamma
    push!(hyps, gamma)

    λrange = [0.0, 1.0]
    λ = rand(rng, Uniform(λrange[1], λrange[2]))
    logp += log(1.0 / (λrange[2] - λrange[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, λ)

    clip_range = [0.1, 0.3]
    clip_param = rand(rng, Uniform(clip_range[1], clip_range[2]))
    logp += log(1.0 / (clip_range[2] - clip_range[1]))  # logdensity for uniform distribution on [a,b]
    push!(hyps, clip_param)

    entropy_range = [1e-8, 1e-2]
    entropy_coef, elogp = logRand(entropy_range[1], entropy_range[2], rng)
    logp += elogp
    push!(hyps, entropy_coef)

    steps_range = [64, 256]
    steps_per_batch, sblogp = log2RandDisc(steps_range[1], steps_range[2], rng)
    logp += sblogp
    push!(hyps, steps_per_batch)

    epochs_range = [1,10]
    num_epochs = rand(rng, epochs_range[1]:epochs_range[2])
    logp += log(1. / (epochs_range[2] - epochs_range[1] + 1))
    push!(hyps, num_epochs)

    bsize_range = [16, 64]
    batch_size, bslogp = log2RandDisc(bsize_range[1], min(bsize_range[2], steps_per_batch), rng)
    logp += bslogp
    push!(hyps, batch_size)

    ϵ = 1e-5
    push!(hyps, ϵ)

    if discrete_states
        parange = [1e-3, 0.1]
    else
        parange = [1e-6, 1e-3]
    end
    αp, palogp = logRand(parange[1], parange[2], rng)
    logp += palogp
    push!(hyps, αp)

	popt = EvaluationOfRLAlgs.Adam(policy, αp, ϵ=ϵ)
	vopt = EvaluationOfRLAlgs.Adam(vf, αp, ϵ=ϵ)

    agent = EvaluationOfRLAlgs.PPO(policy, vf, popt, vopt, γ, λ, clip_param, entropy_coef, steps_per_batch, num_epochs, batch_size, typeof(env.state), Int)

    return agent, hyps, logp
end


function sample_alg(name, env, rng, num_episodes)

    if name == "actorcritic"
        agent, hyps, logp = CreateActorCritic(env, rng)
    elseif name == "actorcritic-scaled"
        agent, hyps, logp = CreateActorCriticScaled(env, rng)
    elseif name == "actorcritic-parl2"
        agent, hyps, logp = CreateActorCriticParl2(env, rng)
    elseif name == "nactd"
        agent, hyps, logp = CreateNACTD(env, rng)
    elseif name == "sarsalambda"
        agent, hyps, logp = CreateSarsaLambda(env, rng)
    elseif name == "sarsalambda-scaled"
        agent, hyps, logp = CreateSarsaLambdaScaled(env, rng)
    elseif name == "sarsa-parl2"
        agent, hyps, logp = CreateSarsaParl2(env, rng)
    elseif name == "qlambda"
        agent, hyps, logp = CreateQLambda(env, rng)
    elseif name == "qlambda-scaled"
        agent, hyps, logp = CreateQLambdaScaled(env, rng)
    elseif name == "q-parl2"
        agent, hyps, logp = CreateQParl2(env, rng)
    elseif name == "ppo"
        agent, hyps, logp = CreatePPO(env, rng)
    else
        println("algorithm ", name, " not yet supported")
    end

    return agent, hyps, logp
end
