using RecipesBase

struct MountainCarParams{T} <:Any where {T<:Real}
    ucoeff::T # action coefficient for acceleration
    g::T # gravity coeff
    h::T # cosine frequency parameter

    MountainCarParams() = new{Float64}(0.001, 0.0025, 3.)
    MountainCarParams(T::Type) = new{T}(T(0.001), T(0.0025), T(3.))
	MountainCarParams(T::Type, ucoeff, g, h) = new{T}(T(ucoeff), T(g), T(h))
end



mutable struct MountainCar{T,TA} <: AbstractEnvironment where {T<:Real,TA<:Real}
    state::Array{T,1}
    reward::T
    done::Bool
    params::MountainCarParams{T}
    t::T # time
	deterministic_start::Bool

    function MountainCar(T::Type=Float64, TA::Type=Int, deterministic_start::Bool=false)
        new{T,TA}(zeros(T,2), T(0.0), false,
           MountainCarParams(T), T(0.), deterministic_start
       )
    end

	function MountainCar(::Type{T}, ::Type{TA}, deterministic_start::Bool, params::MountainCarParams{T}) where {T, TA}
        new{T,TA}(zeros(T,2), T(0.0), false,
           params, T(0.), deterministic_start
       )
    end
end

function get_gamma(env::MountainCar{T})::T where {T}
	return T(1.0)
end

function randomize_env(::Type{MountainCar{T,TA}}, rng::AbstractRNG, deterministic_start::Bool=false) where {T, TA}
	ubase = 0.001
	u = rand(rng, Uniform(0.8*ubase, 1.2*ubase))  # coefficient for action force
	# fix these to keep the problem around the same difficulty
	g = 0.0025   	# "gravity" parameter
	h = 3.  		# mountain frequency parameter  (fixed because it changes the state width. Need to recompute xlims)

	params = MountainCarParams(T, u, g, h)
	return MountainCar(T, TA, deterministic_start, params)
end

function get_state_desc(env::MountainCar{T})::Array{T,2} where {T<:Real}
    ranges = zeros(T, (2,2))
    ranges[1,:] .= [-1.2, 0.5]       # x range
	ranges[2,:] .= [-0.07, 0.07]     # xDot range
	# ranges[3,:] .= [0., 5000.]       # time range
    return ranges
end

function get_action_desc(env::MountainCar{T,TA})::Int where{T<:Real,TA<:Integer}
    return 3
end

function get_action_desc(env::MountainCar{T,TA})::Array{TA,1} where{T<:Real,TA<:AbstractFloat}
	ranges = zeros(TA, (1,2))
	ranges[1, 1] = -1.
	ranges[1, 2] = 1.
	return ranges
end

function mountaincar_sim!(state, u, params)
	x, xDot = state
	xDot = xDot + params.ucoeff * u - params.g * cos(params.h * x)
	xDot = clamp(xDot, -0.07, 0.07)
	x += xDot

	if x < -1.2
		x = -1.2
		xDot = 0.
	end

	state[1:2] .= x, xDot
end

function step!(env::MountainCar{T}, action::Int, rng::AbstractRNG)::T where {T<:Real}
    if action <= 0 || action > 3
        error("Action needs to be an integer in [1, 3]")
    end
    u::T = 0.
	u = T(action) - 2.

	mountaincar_sim!(env.state, u, env.params)
	env.t += 1
	# env.state[3] = env.t

    env.reward = -1.0
	env.done = terminalCheck(env)

    return env.reward
end

function terminalCheck(env::MountainCar)::Bool
	goalcond = env.state[1] ≥ 0.5
	timecond = env.t ≥ 5000
	done = goalcond | timecond

	return done
end

function step!(env::MountainCar{T}, action::T, rng::AbstractRNG)::T where {T<:AbstractFloat}
	u = clamp(action, -1., 1.)

	mountaincar_sim!(env.state, u, env.params)
	env.t += 1
	# env.state[3] = env.t

    env.reward = -1.0
	env.done = terminalCheck(env)

    return env.reward
end

function get_state(env::MountainCar{T})::Array{T,1} where {T<:Real}
    return env.state
end

function get_reward(env::MountainCar)
    return env.reward
end

function is_terminal(env::MountainCar)::Bool
    return env.done
end

function env_reset!(env::MountainCar, rng)
	env.state[1] = -0.5
	env.state[2] = 0.
	# env.state[3] = 0.
	if !env.deterministic_start
		env.state[1] = rand(rng, Uniform(-0.6, -0.4))
	end
    env.t = 0.
	env.done = false
end

function clone(env::MountainCar{T,TA}) where {T<:Real,TA<:Union{Int,<:AbstractFloat}}
	env2 = MountainCar(T,TA, env.deterministic_start)
	env2.state .= deepcopy(env.state)
	env2.reward = deepcopy(env.reward)
	env2.done = deepcopy(env.done)
	env2.t = deepcopy(env.t)
	env2.params = deepcopy(env.params)
	return env2
end


@recipe function f(env::MountainCar)
	x, xvel = env.state
	h = env.params.h
	y = sin(h * x)

	xpts = range(-1.2, stop=0.5, length=50)
	ypts = map(x->sin(h*x), xpts)

	legend := false
	xlims := (-1.2, 0.5)
	ylims := (min(ypts...), max(ypts...))
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

	# mountain
	@series begin
		seriestype := :line
		linecolor := :black
		linewidth := 2

		xpts, ypts
	end

	# car
	@series begin
		seriestype := :shape
		linecolor := nothing
		color := :blue
		aspect_ratio := 1.

		circleShape(x, y, 0.05)
	end
end
