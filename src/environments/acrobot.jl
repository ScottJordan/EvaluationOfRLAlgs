using RecipesBase

struct AcrobotParams{T} <:Any where {T<:Real}
    m1::T 	# mass of first link
    m2::T 	# mass of second link
    l1::T 	# length of first link
	l2::T 	# length of second link
	lc1::T 	# position of center mass of link1
	lc2::T 	# position of center mass of link2
	i1::T  	# link1 moment of inertia
	i2::T  	# link2 moment of inertia
	g::T   	# gravity force (not directional)
	fmag::T # max force magnitude

    AcrobotParams() = new{Float64}(1., 1., 1., 1., 0.5, 0.5, 1., 1., 9.8, 1.)
    AcrobotParams(T::Type) = new{T}(1., 1., 1., 1., 0.5, 0.5, 1., 1., 9.8, 1.)
	AcrobotParams(T::Type, m1, m2, l1, l2, g) = new{T}(m1, m2, l1, l2, 0.5*l1, 0.5*l2, 1., 1., g, 1.)
end

mutable struct Acrobot{T,TA} <: AbstractEnvironment where {T<:Real,TA<:Real}
    state::Array{T,1}
    reward::T
    done::Bool
    params::AcrobotParams{T}
	dt::T
	integShritte::T  # larger this is more accurate Runge-Kutta approximation
    t::T # time
	deterministic_start::Bool

    function Acrobot(T::Type=Float64, TA::Type=Int, deterministic_start::Bool=false)
        new{T,TA}(zeros(T,4), T(0.0), false,
           AcrobotParams(T), T(0.2), T(10.), T(0.), deterministic_start
       )
    end

	function Acrobot(::Type{T}, ::Type{TA}, deterministic_start::Bool, params::AcrobotParams{T}) where {T, TA}
        new{T,TA}(zeros(T,4), T(0.0), false,
           params, T(0.2), T(10.), T(0.), deterministic_start
       )
    end
end

function get_gamma(env::Acrobot{T})::T where {T}
	return T(1.0)
end

function randomize_env(::Type{Acrobot{T,TA}}, rng::AbstractRNG, deterministic_start::Bool=false) where {T, TA}
	m1 = rand(rng, Uniform(0.8, 1.2)) 	# mass of link 1
	l1 = rand(rng, Uniform(0.8, 1.2))   # length of link 1
	m2 = rand(rng, Uniform(0.8, 1.2))  	# mass of link 2
	l2 = rand(rng, Uniform(0.8, 1.2))   # length of link 2
	g = rand(rng, Uniform(8., 10.))     # gravity
	params = AcrobotParams(T, m1, m2, l1, l2, g)
	return Acrobot(T, TA, deterministic_start, params)
end

function get_state_desc(env::Acrobot{T})::Array{T,2} where {T<:Real}
    ranges = zeros(T, (4,2))
    ranges[1,:] .= [-π, π]           # theta1 range
	ranges[2,:] .= [-π, π]           # theta2 range
	ranges[3,:] .= [-4. * π, 4. * π]       # theta1Dot range
	ranges[4,:] .= [-9. * π, 9. * π]       # theta2Dot range
    return ranges
end

function get_action_desc(env::Acrobot{T,TA})::Int where{T<:Real,TA<:Integer}
    return 3
end

function get_action_desc(env::Acrobot{T,TA})::Array{TA,1} where{T<:Real,TA<:AbstractFloat}
	ranges = zeros(TA, (1,2))
	ranges[1, 1] = -1.
	ranges[1, 2] = 1.
	return ranges
end

function rk_helper!(buff::Array{T, 1}, s::Array{T, 1}, tau::T, params::AcrobotParams{T}) where {T}
	m1, m2, l1, l2, lc1, lc2, i1, i2, g = params.m1, params.m2, params.l1, params.l2, params.lc1, params.lc2, params.i1, params.i2, params.g

	d1 = m1*lc1^2 + m2*(l1^2 + lc2^2 + 2. * l1*lc2*cos(s[2])) + i1 + i2
	d2 = m2*(lc2^2 + l1*lc2*cos(s[2])) + i2

	phi2 = m2*lc2*g*cos(s[1] + s[2] - (π / 2.))
	phi1 = (-m2*l1*lc2*s[4]^2 * sin(s[2]) - 2. * m2*l1*lc2*s[4]*s[3] * sin(s[2]) + (m1*lc1 + m2*l1)*g*cos(s[1] - (π / 2.)) + phi2)

	newa2 = ((1. / (m2*lc2^2 + i2 - (d2^2) / d1)) * (tau + (d2 / d1)*phi1 - m2*l1*lc2*s[3]^2 * sin(s[2]) - phi2))
	newa1 = ((-1. / d1) * (d2*newa2 + phi1))
	buff[1] = s[3]
	buff[2] = s[4]
	buff[3] = newa1
	buff[4] = newa2
end

function Acrobot_sim!(state::Array{T,1}, u::T, params::AcrobotParams{T}, dt, integShritte) where {T}
	theta1, theta2, theta1Dot, theta2Dot = state
	hilf = deepcopy(state)
	s0_dot = zeros(T, 4)
	s1_dot = zeros(T, 4)
	s2_dot = zeros(T, 4)
	s3_dot = zeros(T, 4)
	ss = zeros(T, 4)
	s1 = zeros(T, 4)
	s2 = zeros(T, 4)
	s3 = zeros(T, 4)

	h = dt / integShritte

	for i in 1:integShritte
		rk_helper!(s0_dot, hilf, u, params)
		@. s1 = hilf + (h / 2.) * s0_dot

		rk_helper!(s1_dot, s1, u, params)
		@. s2 = hilf + (h / 2.) * s1_dot

		rk_helper!(s2_dot, s2, u, params)
		@. s3 = hilf + (h / 2.) * s2_dot

		rk_helper!(s3_dot, s3, u, params)
		@. hilf = hilf + (h / 6.) * (s0_dot + 2. * (s1_dot + s2_dot) + s3_dot)
	end

	@. ss = hilf

	theta1 = mod(ss[1] + π, 2. * π) - π
	theta2 = mod(ss[2] + π, 2. * π) - π

	theta1Dot = clamp(ss[3], -4. * π, 4. * π)
	theta2Dot = clamp(ss[4], -9. * π, 9. * π)

	state .= theta1, theta2, theta1Dot, theta2Dot

end

function step!(env::Acrobot{T}, action::Int, rng::AbstractRNG)::T where {T<:Real}
    if action <= 0 || action > 3
        error("Action needs to be an integer in [1, 3]")
    end
    u::T = 0.
	u = (T(action) - 2.) * env.params.fmag

	Acrobot_sim!(env.state, u, env.params, env.dt, env.integShritte)
	env.t += env.dt

    env.reward = -0.1
	env.done = terminalCheck(env)
	if env.done
		env.reward = 0.
	end

	return env.reward
end

function terminalCheck(env::Acrobot)::Bool
	elbowY = -env.params.l1*cos(env.state[1])
	handY = elbowY - env.params.l2*cos(env.state[1] + env.state[2])
	anglecond = handY > env.params.l1
	timecond = env.t ≥ 400.
	done = anglecond | timecond

	return done
end

function step!(env::Acrobot{T}, action::T, rng::AbstractRNG)::T where {T<:AbstractFloat}
	u = clamp(action, -1., 1.) * env.params.fmag

	Acrobot_sim!(env.state, u, env.params, env.dt, env.integShritte)
	env.t += env.dt

    env.reward = -0.1
	env.done = terminalCheck(env)
	if env.done
		env.reward = 0.
	end
end

function get_state(env::Acrobot{T})::Array{T,1} where {T<:Real}
    return env.state
end

function get_reward(env::Acrobot)
    return env.reward
end

function is_terminal(env::Acrobot)::Bool
    return env.done
end

function env_reset!(env::Acrobot, rng)
	fill!(env.state, 0.)
	if !env.deterministic_start
		env.state[1:2] = rand(rng, Uniform(-π * 5. / 180., π * 5. / 180.), 2)
	end
    env.t = 0.
	env.done = false
end

function clone(env::Acrobot{T,TA}) where {T<:Real,TA<:Union{Int,<:AbstractFloat}}
	env2 = Acrobot(T,TA, env.deterministic_start)
	env2.state .= deepcopy(env.state)
	env2.reward = deepcopy(env.reward)
	env2.done = deepcopy(env.done)
	env2.t = deepcopy(env.t)
	env2.params = deepcopy(env.params)
	env2.dt = env.dt
	env2.integShritte = env.integShritte
	return env2
end





@recipe function f(env::Acrobot)
	theta1, theta2 = env.state[1:2]
	l1, l2 = env.params.l1, env.params.l2
	maxlen = (l1+l2)*1.2
	x1,y1 = 0., 0.
	x2 = x1 - sin(theta1)*l1
	y2 = y1 - cos(theta1)*l1
	x3 = x2 - sin(theta1 + theta2)*l2
	y3 = y2 - cos(theta1 + theta2)*l2

	legend := false
	xlims := (-maxlen, maxlen)
	ylims := (-maxlen, maxlen)
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

	# target line
	@series begin
		linecolor := :red
		linewidth := 5

		[0., 0.], [0., maxlen]
	end

	# arms
	@series begin
		linecolor := :black
		linewidth := 10

		[x1, x2, x3], [y1, y2, y3]
	end

end
