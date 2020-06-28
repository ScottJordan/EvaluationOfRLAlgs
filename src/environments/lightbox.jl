mutable struct LightBox{T,Int} <: AbstractEnvironment where {T}
    state::Array{T, 1}
    reward::Float64
    done::Bool
    time_step::Int
	stochastic::Bool
	randchance::T

function LightBox(::Type{T}, stochastic::Bool=true, chance::Float64=0.9) where{T}
        state = zeros(T, 20)
        new{T,Int}(state, 0., false, 0, stochastic, chance)
    end
end


function get_gamma(env::LightBox)
	return 1.0
end


function get_state_desc(env::LightBox{T})::Array{T,2} where {T}
    ranges = zeros(T, (20, 2))
    ranges[:, 2] .= T(1.)
	return ranges
end

function get_action_desc(env::LightBox)::Int
    return 20
end

function step!(env::LightBox, action::Int, rng::AbstractRNG)
    env.time_step += 1
    if action <= 0 || action > 20
        error("Action needs to be an integer in [1, 20]")
    end

    #values must be on for key to be on
    dependencies = Dict{Int,Tuple}(9=>(0, 3, 6), 10=>(1, 4), 11=>(2, 5),
                    12=>(2, 1), 13=>(5, 4), 14=>(8, 7, 6),
                    15=>(9, 10), 16=>(10, 11), 17=>(12, 13),
                    18=>(13, 14), 19=>(16, 17))

    #key must be on for corresponding values to be on
    causal = Dict{Int,Tuple}(0=>(0, 9), 1=>(1, 10, 12), 2=>(2, 11, 12), 3=>(3, 9),
              4=>(4,10,13), 5=>(5, 11, 13), 6=>(6, 9, 14), 7=>(7, 14),
              8=>(8, 14), 9=>(9, 15), 10=>(10, 15, 16), 11=>(11, 16),
	         12=>(12, 17), 13=>(13, 17, 18), 14=>(14, 18), 15=>(15,),
             16=>(16, 19), 17=>(17, 19), 18=>(18,), 19=>(19,))



    if !env.stochastic || rand(rng) < env.randchance
        # update state
        if is_legal_move(env.state, action, dependencies)
            env.state[action] = 1.0 - env.state[action]
            update_lights!(env.state, action, causal)
        else
            env.state .= 0
        end
	end

    env.reward = -1.0
	max_steps = 5000
    env.done = env.time_step ≥ max_steps
    if isapprox(env.state[end], 1.0)
        env.reward = 1.0
        env.done = true
    end
    return env.reward
end

function is_legal_move(state, action, dependencies)
    if isapprox(state[action], 1.) || action < 10
        return true #always legal to turn off a light and toggle a base light
    end

    for i in dependencies[action-1]
        if isapprox(state[i+1], 0.)
            return false
        end
    end
    return true
end

function update_lights!(state, action, causal)
    #if light is off, turn off all dependent lights
    if isapprox(state[action], 0.)
        for i in causal[action-1][2:end]
            state[i+1] = 0.
            update_lights!(state, i+1, causal)
        end
    end
end

function get_state(env::LightBox)
    return env.state
end

function get_reward(env::LightBox)::Float64
    return env.reward
end

function is_terminal(env::LightBox)::Bool
    return env.done
end

function env_reset!(env::LightBox, rng)
    env.time_step = 0
    env.state .= 0.0
    env.done = false
end

function clone(env::LightBox)::LightBox
    env2 = LightBox(env.stochastic, env.randchance)
    env2.time_step = deepcopy(env.time_step)
    env2.state = deepcopy(env.state)
    env2.done = deepcopy(env.done)
    env2.reward = deepcopy(env.reward)
    return env2
end
