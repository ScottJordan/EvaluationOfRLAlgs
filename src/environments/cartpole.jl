using RecipesBase

struct CartPoleParams{T} <:Any where {T<:Real}
    m::T # mass of pole
    l::T # length of pole
    mc::T # mass of cart
    muc::T # some constant?
    mup::T # some constant?
    fmag::T # magnitude of force applied
	g::T  # gravity (not signed)
    CartPoleParams() = new{Float64}(0.1, 0.5, 1., 0.0005, 0.000002, 10., 9.8)
    CartPoleParams(T::Type) = new{T}(T(0.1), T(0.5), T(1.), T(0.0005), T(0.000002), T(10.), T(9.8))
	CartPoleParams(T::Type, m, l, mc, g) = new{T}(m, l, mc, T(0.0005), T(0.000002), T(10.), g)
end



mutable struct CartPole{T,TA} <: AbstractEnvironment where {T<:Real,TA<:Real}
    state::Array{T,1}
    reward::T
    done::Bool
    params::CartPoleParams{T}
    dt::T
    simSteps::Int
    t::T # time

    function CartPole(T::Type=Float64, TA::Type=Int)
        new{T,TA}(zeros(T,4), T(0.0), false,
           CartPoleParams(T), T(0.02), 1, T(0.)
       )
    end

	function CartPole(::Type{T}, ::Type{TA}, params::CartPoleParams{T}) where {T,TA}
		new{T,TA}(zeros(T,4), T(0.0), false,
		   params, T(0.02), 1, T(0.)
	   )
	end

end

function get_gamma(env::CartPole{T})::T where {T}
	return T(1.0)
end


function randomize_env(::Type{CartPole{T,TA}}, rng::AbstractRNG) where {T, TA}
	m = rand(rng, Uniform(0.025, .25))  # mass of pole
	l = rand(rng, Uniform(0.1, 1.0))    # length of pole
	mc = rand(rng, Uniform(0.1, 5.))    # mass of cart
	g = rand(rng, Uniform(8., 10.))     # gravity
	params = CartPoleParams(T, m, l, mc, g)
	return CartPole(T, TA, params)
end

function get_state_desc(env::CartPole{T})::Array{T,2} where {T<:Real}
    ranges = zeros(T, (4,2))
    ranges[1,:] .= [-2.4, 2.4]       # x range
	ranges[2,:] .= [-10., 10.]#[-6, 6.]       # xDot range
	ranges[3,:] .= [-π / 12.0, π / 12.0] # theta range
	ranges[4,:] .= [-π, π]           # thetaDot range
	# ranges[5,:] .= [0., 20.]
    return ranges
end

function get_action_desc(env::CartPole{T,TA})::Int where{T<:Real,TA<:Integer}
    return 2
end

function get_action_desc(env::CartPole{T,TA}) where{T<:Real,TA<:AbstractFloat}
	ranges = zeros(TA, (1,2))
	ranges[1, 1] = -1.
	ranges[1, 2] = 1.
	return ranges
end

function cartpole_sim(state::Array{T,1}, constants::CartPoleParams{T}, u::T, steps::Int, dt::T) where {T<:Real}
	x, xDot, theta, thetaDot = view(state, 1:4)
	omegaDot::T = 0.
	vDot::T = 0.
	m = constants.m     # mass of pole
    l = constants.l     # length of pole
    mc = constants.mc   # mass of cart
    muc = constants.muc # some constant?
    mup = constants.mup # some constant?
	g = constants.g
	# for i in 1:steps
	omegaDot = (g * sin(theta) + cos(theta) * (muc * sign(xDot) - u - m * l * thetaDot^2 * sin(theta)) / (m + mc) - mup * thetaDot / (m * l)) / (l * (4.0 / 3.0 - m / (m + mc) * cos(theta)^2))
    vDot = (u + m * l * (thetaDot^2 * sin(theta) - omegaDot*cos(theta)) - muc*sign(xDot)) / (m + mc)
    theta += dt * thetaDot
    thetaDot += dt * omegaDot
    x += dt * xDot
    xDot += dt * vDot

    theta = mod(theta + π, 2 * π) - π
	# end
	state[1:4] .= x, xDot, theta, thetaDot
end

function step!(env::CartPole{T}, action::Int, rng::AbstractRNG)::T where {T<:Real}
    if action <= 0 || action > 2
        error("Action needs to be an integer in [1, 2]")
    end
    u::T = 0.
    if action == 1
        u = -env.params.fmag
    else
        u =  env.params.fmag
    end

	cartpole_sim(env.state, env.params, u, env.simSteps, env.dt/env.simSteps)
	env.t += env.dt
	# env.state[5] = env.t
    # env.state[1] = clamp(env.state[1], -2.4, 2.4)

    env.reward = 1.0
	env.done = terminalCheck(env)
	# if done
	# 	println(env.t, env.dt)
	# end

    return env.reward
end

function terminalCheck(env::CartPole)::Bool
	polecond = abs(env.state[3]) > (π / 15.0)
	cartcond = abs(env.state[1]) ≥ 2.4
	timecond = env.t ≥ (20. - 1e-8)
	done = polecond | cartcond | timecond
	# if done
	# 	println([polecond, cartcond, timecond])
	# end
	return done
	# return (abs(env.state[3]) > (π / 15.0)) | (abs(env.state[1]) ≥ 2.4) | (env.t ≥ (20. - 1e-8))
end

function step!(env::CartPole{T}, action::T, rng::AbstractRNG)::T where {T<:AbstractFloat}
	u = clamp(action, -1., 1.)
	u *= env.params.fmag

	cartpole_sim(env.state, env.params, u, env.simSteps, env.dt/env.simSteps)
	env.t += env.dt
    # env.state[1] = clamp(env.state[1], -2.4, 2.4)
	# env.state[5] = env.t
    env.reward = 1.0
	env.done = terminalCheck(env)

    return env.reward
end

function step!(env::CartPole{T}, action::Array{T}, rng::AbstractRNG)::T where {T<:AbstractFloat}
	u = clamp(action[1], -1., 1.)
	u *= env.params.fmag

	cartpole_sim(env.state, env.params, u, env.simSteps, env.dt/env.simSteps)
	env.t += env.dt
    # env.state[1] = clamp(env.state[1], -2.4, 2.4)
	# env.state[5] = env.t
    env.reward = 1.0
	env.done = terminalCheck(env)

    return env.reward
end

function get_state(env::CartPole{T})::Array{T,1} where {T<:Real}
    return env.state # VectorState([env.x, env.xDot, env.theta, env.thetaDot])
end

function get_reward(env::CartPole)
    return env.reward
end

function is_terminal(env::CartPole)::Bool
    return env.done
end

function env_reset!(env::CartPole, rng)
	fill!(env.state, 0.)
    env.t = 0.
	env.done = false
end

function clone(env::CartPole{T,TA}) where {T<:Real,TA<:Union{Int,<:AbstractFloat}}
	env2 = CartPole(T,TA)
	env2.state .= deepcopy(env.state)
	env2.reward = deepcopy(env.reward)
	env2.done = deepcopy(env.done)
	env2.t = deepcopy(env.t)
	env2.dt = deepcopy(env.dt)
	env2.simSteps = deepcopy(env.simSteps)
	env2.params = deepcopy(env.params)
	return env2
end

# CartPole Rendering originally from JuliaML/Reinforce.jl and has only be modified slighlty
# https://github.com/JuliaML/Reinforce.jl/blob/master/src/envs/cartpole.jl
@recipe function f(env::CartPole)
	x, xvel, θ, θvel = env.state[1:4]
	legend := false
	xlims := (-2.4, 2.4)
	l = env.params.l
	ylims := (-Inf, 2l)
	grid := false
	ticks := nothing

	# pole
	@series begin
		linecolor := :red
		linewidth := 10
		[x, x + 2l * sin(θ)], [0.0, 2l * cos(θ)]
	end

	# cart
	@series begin
		seriescolor := :black
		seriestype := :shape
		hw = 0.5
		l, r = x-hw, x+hw
		t, b = 0.0, -0.1
		[l, r, r, l], [t, t, b, b]
	end
end
