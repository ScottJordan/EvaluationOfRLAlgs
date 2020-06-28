using Random

mutable struct Chain{Int,Int} <: AbstractEnvironment
    state::Int
    reward::Float64
    done::Bool
    size::Int
    time_step::Int
	stochastic::Bool
	randchance::Float64
    function Chain(size::Int, stochastic::Bool=false, chance::Float64=0.2)
        new{Int,Int}(1, 0., false, size, 0, stochastic, chance)
    end
end

function get_gamma(env::Chain)
	return 1.0
end

function randomize_env(::Type{T}, rng::AbstractRNG, size::Int, stochastic::Bool) where {T <: Chain}
	ϵ = rand(rng, Uniform(0., 0.25))
	return Chain(size, stochastic, ϵ)
end

function get_state_desc(env::Chain)::Int
    return env.size
end

function get_action_desc(env::Chain)::Int
    return 2
end

function step!(env::Chain, action::Int, rng::AbstractRNG)::Float64
    env.time_step += 1
    if action <= 0 || action > 2
        error("Action needs to be an integer in [1, 2]")
    end

    if env.stochastic
		tmp = rand(rng)
		if tmp < env.randchance / 2.  # if random 50% chance to not move
			action = 0
		elseif tmp < env.randchance  # if random 50% chance to move in oposite direction
			action = 3 - action
		end
	end

    if action == 1
        env.state -= 1
    elseif action == 2
        env.state += 1
    end

	env.state = clamp(env.state, 1, env.size)

    env.reward = -1.0
	max_steps = 20 * env.size#^2
    env.done = (env.state == env.size) | (env.time_step ≥ max_steps)

	if env.done
		env.reward = 0.
	end

    return env.reward
end

function get_state(env::Chain)::Int
    return env.state
end

function get_reward(env::Chain)::Float64
    return env.reward
end

function is_terminal(env::Chain)::Bool
    return env.done
end

function env_reset!(env::Chain, rng)
	env.time_step = 0
    env.state = 1
    env.done = false
end

function clone(env::Chain)::Chain
    env2 = Chain(env.size, env.stochastic, env.randchance)
    env2.time_step = deepcopy(env.time_step)
    env2.state = deepcopy(env.state)
    env2.done = deepcopy(env.done)
    env2.reward = deepcopy(env.reward)
    return env2
end
