using Random; import Future
using Base
using Plots


include("rendering.jl")

function run_agent!(env::TE, agent::TA, num_episodes::Int, rng::AbstractRNG)::Array{Float64, 1} where {TE<:AbstractEnvironment,TA<:AbstractAgent}
    returns = zeros(num_episodes)
    step::Int = 0
    for episode in 1:num_episodes
        env_reset!(env, rng)
        new_episode!(agent, rng)
        step = 0

        while !is_terminal(env)
            act!(agent, env, rng)
            returns[episode] += env.reward
        end
    end
    return returns
end

function eval_agent(env::TE, agent::TA, num_episodes::Int, rng::AbstractRNG)::Array{Float64, 1} where {TE<:AbstractEnvironment,TA<:AbstractAgent}
    returns = zeros(num_episodes)
    step::Int = 0
    for episode in 1:num_episodes
        env_reset!(env, rng)
        new_episode!(agent, rng)
        step = 0

        while !is_terminal(env)
            act!(agent, env, rng, false)  # set training flag to false
            returns[episode] += env.reward
        end
    end
    return returns
end

function eval_policy(env::TE, policy, γ, num_episodes::Int, rng::AbstractRNG)::Array{Float64, 1} where {TE<:AbstractEnvironment}
    returns = zeros(num_episodes)
    step::Int = 0
    for episode in 1:num_episodes
        env_reset!(env, rng)
        # new_episode!(agent, rng)
        step = 0

        while !is_terminal(env)
            get_action!(policy, env.state, rng)
            r = step!(env, policy.action, rng)
            returns[episode] += γ^step * r
            step += 1
        end
    end
    return returns
end

function run_agent!(env::AbstractEnvironment, agent::AbstractAgent, num_episodes::Int, rng::AbstractRNG, draw::Bool)::Array{Float64, 1}
    returns = zeros(num_episodes)
    step::Int = 0
    for episode in 1:num_episodes
        env_reset!(env, rng)
        new_episode!(agent, rng)
        step = 0
        if draw
            plt = plot(env)
            display(plt);
            gui()
        end

        while !is_terminal(env)
            act!(agent, env, rng)
            returns[episode] += env.reward
            if draw
                plt = plot(env)
                display(plt);
                gui()
            end
        end
    end
    return returns
end


function run_agent(env::AbstractEnvironment, agent::AbstractAgent, num_trials::Int, num_episodes::Int, rng::MersenneTwister)::Array{Float64, 2}
    returns = zeros(num_episodes, num_trials)
    agents = [clone(agent) for i in 1:num_trials]
    envs = [clone(env) for i in 1:num_trials]

    for trial in 1:num_trials
        for episode in 1:num_episodes
            env_reset!(envs[trial], rng)
            new_episode!(agents[trial], rng)

            while !is_terminal(envs[trial])
                act!(agents[trial], envs[trial], rng)
                returns[episode, trial] += envs[trial].reward
            end
        end
    end

    return returns
end

function run_agent_parallel(env::AbstractEnvironment, agent::AbstractAgent, num_trials::Int, num_episodes::Int, rng::MersenneTwister)::Array{Float64, 2}
    agents = [clone(agent) for i in 1:num_trials]
    envs = [clone(env) for i in 1:num_trials]

    rngs = [rng; accumulate(Future.randjump, fill(big(10)^20, num_trials-1), init=rng)]

    returns = zeros(num_episodes, num_trials)

    Threads.@threads for trial in 1:num_trials
        for episode in 1:num_episodes
            env_reset!(envs[trial], rngs[trial])
            new_episode!(agents[trial], rngs[trial])

            while !is_terminal(envs[trial])
                act!(agents[trial], envs[trial], rngs[trial])
                returns[episode, trial] += envs[trial].reward
            end
        end
    end

    return returns
end
