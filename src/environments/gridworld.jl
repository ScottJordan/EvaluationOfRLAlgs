using Random

mutable struct GridWorld{Int,Int} <: AbstractEnvironment
    state::Int
    reward::Float64
    done::Bool
    size::Int
    pos::Tuple{Int, Int}#Array{Int, 1}
    time_step::Int
	stochastic::Bool
	randchance::Float64
    function GridWorld(size::Int, stochastic::Bool=false, chance::Float64=0.2)
        new{Int,Int}(1, 0., false, size, (1,1), 0, stochastic, chance)
    end
end

function get_gamma(env::GridWorld)
	return 1.0
end

function randomize_env(::Type{T}, rng::AbstractRNG, size::Int, stochastic::Bool) where {T <: GridWorld}
	ϵ = rand(rng, Uniform(0., 0.25))
	return GridWorld(size, stochastic, ϵ)
end

function get_state_desc(env::GridWorld)::Int
    return env.size^2
end

function get_action_desc(env::GridWorld)::Int
    return 4
end

function step!(env::GridWorld, action::Int, rng::AbstractRNG)::Float64
    env.time_step += 1
    if action <= 0 || action > 4
        error("Action needs to be an integer in [1, 4]")
    end
    x,y = env.pos

    if env.stochastic
		temp = rand(rng)
		noeffect = 1. - env.randchance  # e.g., 0.8
		stay = 1. - (env.randchance / 2.) # e.g., 0.9
		side = 1. - (env.randchance / 4.) # e.g., 0.95
		if (temp < noeffect)  # Take proposed action
			# do nothing
		elseif (temp < stay)
			action = 0    # Stay put. Encode as action -1
		elseif (temp < side)
			action += 1
			if (action == 5)
				action = 1
			end
		else
			action -= 1
			if (action == 0)
				action = 4
			end
		end
	end

    if action == 1
        y -= 1
    elseif action == 2
        y += 1
    elseif action == 3
        x -= 1
    elseif action == 4
        x += 1
    end
    x = clamp(x, 1, env.size)
    y = clamp(y, 1, env.size)
    env.pos = (x, y)
    env.reward = -1.0
    env.state = (x-1)*env.size + y
	max_steps = 20 * env.size^2
    env.done = (env.pos == (env.size, env.size)) | (env.time_step ≥ max_steps)

    return env.reward
end

function get_state(env::GridWorld)::Int
    return env.state
end

function get_reward(env::GridWorld)::Float64
    return env.reward
end

function is_terminal(env::GridWorld)::Bool
    return env.done
end

function env_reset!(env::GridWorld, rng)
    env.pos = (1,1)
    env.time_step = 0
    env.state = (env.pos[1]-1)*env.size + (env.pos[2])
    env.done = false
end

function clone(env::GridWorld)::GridWorld
    env2 = GridWorld(env.size, env.stochastic, env.randchance)
    env2.pos = deepcopy(env.pos)
    env2.time_step = deepcopy(env.time_step)
    env2.state = deepcopy(env.state)
    env2.done = deepcopy(env.done)
    env2.reward = deepcopy(env.reward)
    return env2
end
