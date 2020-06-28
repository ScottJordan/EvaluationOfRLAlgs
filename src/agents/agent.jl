include("actorcritic.jl")
include("reinforce.jl")
include("sarsa.jl")
include("q.jl")
include("fdpolicysearch.jl")
include("nactd.jl")
include("ppo.jl")

export ActorCritic, REINFORCE, NACTD
export Sarsa, ExpectedSarsa, QLearning
export PPO
export AbstractFiniteDifferencePolicySearch, BasicRandomSearch, AugmentedRandomSearch


function act!(agent::TA, env::TE, rng::AbstractRNG, learn::Bool) where {TA<:AbstractPolicyAgent,TE<:AbstractEnvironment} # do not learn if learn flag is set
    logp = get_action!(agent.π, env.state, rng)
    r = step!(env, agent.π.action, rng)
end
