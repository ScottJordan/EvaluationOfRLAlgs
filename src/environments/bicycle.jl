# Bicycle domain ported from rlpy https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Bicycle.py
# original paper Learning to Drive a Bicycle using Reinforcement Learning and Shaping, Jette Randlov, Preben Alstrom, 1998. https://www.semanticscholar.org/paper/Learning-to-Drive-a-Bicycle-Using-Reinforcement-and-Randl%C3%B8v-Alstr%C3%B8m/10bad197f1c1115005a56973b8326e5f7fc1031c
using RecipesBase

# Bicycle domain might not be correct.
struct BicycleParams{T} <:Any where {T<:Real}
	g::T    	# gravity 9.82
	v::T		# velocity of the bicycle (10 km/h paper value) default is 10. / 3.6 (10kph appox 6.2mph and 10kph is 2.77778 meters per second or 10/3)
	d_CM::T 	# vertical distance between the CM for the bicycle and for the cyclist 0.3m
	c::T    	# horizontal distance between the point where the front wheel touches the ground and the CM 0.66 m
	h::T		# height of the CM over the ground  0.94 cm
	M_c::T  	# mass bicycle  9kg ≈ 19.8lbs 15kg ≈ 33lbs (default is 15kg)
	M_d::T		# mass tire 1.0kg ≈ 2.2lbs 1.5kg ≈ 3.3lbs 3.0kg ≈ 4.4lbs (default is 1.7)
	M_p::T		# mass cyclist M_p = 45.4kg ≈ 100lbs 60.0kg ≈ 132.0lbs 90.7kg ≈ 200lbs (default is 60kg)
	M::T   		# mass of cyclist and bike M_p + M_c
	r::T	   	# radius of tire 0.34m
	dsigma::T  	# angular velcity of a tire (v/r)
	I::T    	# moment of inertia for bicycle and cyclist 13/3 * M_c * h^2 + M_p * (h + d_CM)^2
	I_dc::T 	# moment of inertia for tire M_d * r^2
	I_dv::T 	# moment of inertia for tire 3/2 * M_d * r^2
	I_dl::T 	# moment of inertia for tire 1/2 * M_d * r^2
	l::T    	# disance between the front and tire and the back tire at the point where they both touch the ground 1.11m

    function BicycleParams(T::Type=Float64)
		g = T(9.82)
		v = T(10. / 3.6)
		d_CM = T(0.3)
		c = T(0.66)
		h = T(0.94)
		M_c = T(15.)
		M_d = T(1.7)
		M_p = T(60.)
		M = M_c + M_p
		r = T(0.34)
		dsigma = T(v / r)
		I = T(13. / 3. * M_c * h^2 + M_p * (h+d_CM)^2)
		I_dc = T(M_d * r^2)
		I_dv = T(3. / 2.)
		I_dl = T(M_d / 2. * r^2)
		l = T(1.11)
		new{T}(g, v, d_CM, c, h, M_c, M_d, M_p, M, r, dsigma, I, I_dc, I_dv, I_dl, l)
	end

	function BicycleParams(::Type{T}, g, M_c, M_d, M_p, v) where {T}
		d_CM = T(0.3)
		c = T(0.66)
		h = T(0.94)
		M = M_c + M_p
		r = T(0.34)
		dsigma = T(v / r)
		I = T(13. / 3. * M_c * h^2 + M_p * (h+d_CM)^2)
		I_dc = T(M_d * r^2)
		I_dv = T(3. / 2. * M_d * r^2)
		I_dl = T(M_d / 2. * r^2)
		l = T(1.11)
		new{T}(g, v, d_CM, c, h, M_c, M_d, M_p, M, r, dsigma, I, I_dc, I_dv, I_dl, l)
	end
end

#
#

mutable struct BikeState{T} <: Any where {T}
	omega::T
	omegaDot::T
	theta::T
	thetaDot::T
	psi::T
	x_b::T
	y_b::T

	function BikeState(::Type{T}) where {T}
		new{T}(T(0.), T(0.), T(0.), T(0.), T(0.), T(0.), T(0.))
	end
end

mutable struct Bicycle{T,Int} <: AbstractEnvironment where {T<:Real}
    state::Array{T,1}
    reward::T
    done::Bool
    params::BicycleParams{T}
	bike::BikeState{T}
    dt::T
    t::T # time

    function Bicycle(T::Type=Float64)
        new{T,Int}(zeros(T,5), T(0.0), false,
           BicycleParams(T), BikeState(T), T(0.01), T(0.)
       )
    end

	function Bicycle(::Type{T}, params::BicycleParams{T}) where {T}
        new{T,Int}(zeros(T,5), T(0.0), false,
           params, BikeState(T), T(0.01), T(0.)
       )
    end
end

function get_gamma(env::Bicycle{T})::T where {T}
	return T(1.0)
end

function randomize_env(::Type{Bicycle{T}}, rng::AbstractRNG) where {T}
	g = rand(rng, Uniform(8., 10.))  		# gravity
	v = rand(rng, Uniform(8., 15.)) / 3.6	# velocity of the bicycle mps
	M_c = rand(rng, Uniform(9., 15.))  		# mass bicycle  9kg ≈ 19.8lbs 15kg ≈ 33lbs (default is 15kg)
	M_d = rand(rng, Uniform(1., 3.))	    	# mass tire 1.0kg ≈ 2.2lbs 1.5kg ≈ 3.3lbs 3.0kg ≈ 4.4lbs (default is 1.7)
	M_p = rand(rng, Uniform(45.4, 90.7))		# mass cyclist M_p = 45.4kg ≈ 100lbs 60.0kg ≈ 132.0lbs 90.7kg ≈ 200lbs (default is 60kg)


	params = BicycleParams(T, g, M_c, M_d, M_p, v)
	return Bicycle(T, params)
end

function get_state_desc(env::Bicycle{T})::Array{T,2} where {T<:Real}
    ranges = zeros(T, (5,2))
    ranges[1,:] .= [-π * 12. / 180., π * 12 / 180]
	ranges[2,:] .= [-π, π]
	ranges[3,:] .= [-π * 80. / 180., π * 80. / 180.]
	ranges[4,:] .= [-π, π]
	ranges[5,:] .= [-π, π]
    return ranges
end

function get_action_desc(env::Bicycle{T,TA})::Int where{T<:Real,TA<:Integer}
    return 9
end


function bicycle_sim(state::BikeState{T}, constants::BicycleParams{T}, u::T, d::T, dt::T, rng::AbstractRNG) where {T<:Real}
	omega, omegaDot, theta, thetaDot, psi = state.omega, state.omegaDot, state.theta, state.thetaDot, state.psi
	g = constants.g
	v = constants.v
	d_CM = constants.d_CM
	c = constants.c
	h = constants.h
	M_c = constants.M_c
	M_d = constants.M_d
	M_p = constants.M_p
	M = constants.M
	r = constants.r
	dsigma = constants.dsigma
	I = constants.I
	I_dc = constants.I_dc
	I_dv = constants.I_dv
	I_dl = constants.I_dl
	l = constants.l

	w = rand(rng, Uniform(-0.02, 0.02), 1)[1]


	phi = omega + atan(d+w) / h
	invr_f = abs(sin(theta)) / l
	invr_b = abs(tan(theta)) / l
	if theta != 0.
		invr_CM = ((l-c)^2 + invr_b^(-2))^(-0.5)
	else
		invr_CM = 0.
	end

	nomega = omega + dt * omegaDot
	nomegaDot = omegaDot + dt * (M * h * g * sin(phi) - cos(phi) * (I_dc * dsigma * thetaDot + sign(theta) * v^2 * (M_d * r * (invr_f + invr_b) + M * h * invr_CM))) / I
	out = theta + dt * thetaDot
	if out < (π * 80. / 180.)
		ntheta = out
		nthetaDot = thetaDot + dt * (u - I_dv * dsigma * omegaDot) / I_dl
	else
		ntheta = sign(out) * (π * 80. / 180.)
		nthetaDot = 0.
	end

	npsi = psi + dt * sign(theta) * v * invr_b

	npsi = npsi % (2 * π)
	if npsi > π
		npsi -= 2 * π
	end

	state.omega = nomega
	state.omegaDot = nomegaDot
	state.theta = ntheta
	state.thetaDot = nthetaDot
	state.psi = npsi

	return npsi - psi

end


function step!(env::Bicycle{T}, action::Int, rng::AbstractRNG)::T where {T<:Real}
    if action <= 0 || action > 10
        error("Action needs to be an integer in [1, 9]")
    end
    u::T = 0.
	d::T = 0.
	uaction = action // 3
	daction = action % 3

	if uaction == 1
        u = -2.
    elseif uaction == 2
    	u =  0.
	else
		u = 2.
    end

	if daction == 1
		d = -0.02
	elseif daction == 2
		d =  0.
	else
		d = 0.02
	end

	psi_diff = bicycle_sim(env.bike, env.params, u, d, env.dt, rng)
	update_state!(env)
	env.t += env.dt


    env.reward = -0.1 * psi_diff + 1.0
	env.done = terminalCheck(env)
	# if env.done
	# 	env.reward = -1.
	# end

    return env.reward
end

function terminalCheck(env::Bicycle)::Bool
	omega = env.bike.omega
	fallcond = abs(omega) > (π * 12. / 180.)
	timecond = env.t ≥ (30. - 1e-8)
	done = fallcond | timecond

	return done
end

function update_state!(env::Bicycle)
    env.state .= env.bike.omega, env.bike.omegaDot, env.bike.theta, env.bike.thetaDot, env.bike.psi
end

function get_state(env::Bicycle{T})::Array{T,1} where {T<:Real}
    return env.state
end

function get_reward(env::Bicycle)
    return env.reward
end

function is_terminal(env::Bicycle)::Bool
    return env.done
end

function env_reset!(env::Bicycle, rng)
	fill!(env.state, 0.)
	env.bike.omega = 0.
	env.bike.omegaDot = 0.
	env.bike.theta = 0.0
	env.bike.thetaDot = 0.
	env.bike.psi = 0.
	env.bike.x_b = 0.
	env.bike.y_b = 0.
    env.t = 0.
	env.done = false
end

function clone(env::Bicycle{T})::Bicycle{T} where {T<:Real}
	env2 = Bicycle(T)
	env2.state .= deepcopy(env.state)
	env2.reward = deepcopy(env.reward)
	env2.done = deepcopy(env.done)
	env2.t = deepcopy(env.t)
	env2.dt = deepcopy(env.dt)
	env2.bike = deepcopy(env.bike)
	env2.params = deepcopy(env.params)
	return env2
end
