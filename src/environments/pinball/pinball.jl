# this code was closely ported from the rlpy implementaiton https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Pinball.py
# configs/*.cfg come directly from rlpy

struct PinBallConfig{T}
    start_pos::Tuple{T,T}
    target_pos::Tuple{T,T}
    target_radius::T
    ball_radius::T
    noise::T
    drag::T
    force::T

    function PinBallConfig(::Type{T}, start_pos::Tuple, target_pos::Tuple, target_radius) where {T}
        new{T}(convert.(T, start_pos), convert.(T, target_pos), T(target_radius), T(0.02), T(0.), T(0.995), T(1. / 5.))
    end
    function PinBallConfig(::Type{T}, start_pos::Tuple, target_pos::Tuple, target_radius, ball_radius) where {T}
        new{T}(convert.(T, start_pos), convert.(T, target_pos), T(target_radius), T(ball_radius), T(0.), T(0.995), T(1. / 5.))
    end
    function PinBallConfig(::Type{T}, start_pos::Tuple, target_pos::Tuple, target_radius, ball_radius, noise, drag, force) where {T}
        new{T}(convert.(T, start_pos), convert.(T, target_pos), T(target_radius), T(ball_radius), T(noise), T(drag), T(force))
    end
end

mutable struct BallState{T} <:Any where {T}
    x::T
    y::T
    xDot::T
    yDot::T
    radius::T

    function BallState(position::Tuple{T,T}, radius::T) where {T}
        new{Float64}(Float64(position[1]), Float64(position[2]), 0., 0., Float64(radius))
    end

    function BallState(::Type{T}, position::Tuple{T,T}, radius::T) where {T}
        new{T}(T(position[1]), T(position[2]), T(0.), T(0.), T(radius))
    end
end

struct PinballObstacle{T} <: Any where {T}
    points::Array{Tuple{T,T}, 1}
    minx::T
    miny::T
    maxx::T
    maxy::T

    function PinballObstacle(::Type{T}, points::Array{Tuple{TX, TY}, 1}) where {T,TX,TY}
        pts = [T.(p) for p in points]
        minx, miny = min.(pts...)
        maxx, maxy = max.(pts...)
        new{T}(pts, minx, miny, maxx, maxy)
    end
end

mutable struct PinBall{T,Int} <: AbstractEnvironment where {T<:Real}
    state::Array{T,1}
    reward::T
    done::Bool
    ball::BallState{T}
    obstacles::Array{PinballObstacle{T}, 1}
    config::PinBallConfig{T}
    dt::T
    t::T # time
	deterministic_start::Bool

    function PinBall(T::Type, config::String, deterministic_start::Bool=false)
        obstacles, conf = read_config(T, config)
        ball = BallState(conf.start_pos, conf.ball_radius)
        new{T,Int}(zeros(T,4), T(0.0), false, ball, obstacles, conf, T(1. / 20.), T(0.), deterministic_start)
    end

	function PinBall(config::String, deterministic_start::Bool=false)
		return PinBall(Float64, config, deterministic_start)
	end

	function PinBall(::Type{T}, config::PinBallConfig{T}, deterministic_start::Bool, obstacles::Array{PinballObstacle{T}, 1}, ball::BallState{T}) where {T}
		b = BallState(T, (ball.x, ball.y), ball.radius)
		b.xDot = ball.xDot
		b.yDot = ball.yDot
		new{T,Int}(zeros(T,4), T(0.0), false, b, deepcopy(obstacles), deepcopy(config), T(1. / 20.), T(0.), deterministic_start)
	end
end

function get_gamma(env::PinBall{T})::T where {T}
	return T(1.0)
end

function randomize_env(::Type{PinBall{T}}, rng::AbstractRNG, config::String, deterministic_start::Bool=false) where {T}
	obstacles, conf = read_config(T, config)
	target_radius = conf.target_radius + rand(rng, Uniform(-0.1, 0.1))  * conf.target_radius  # scaled target radius randomly by standard deviation of 10% of the specified radius
    ball_radius = conf.ball_radius + rand(rng, Uniform(-0.1, 0.1)) * conf.ball_radius
    noise = rand(rng, T) * 0.25    # chance for random action
    drag = 1. - exp(rand(rng, Uniform(log(0.001), log(0.1))))  # friction coefficient for ball
    force =  rand(rng, Uniform(0.1, 0.3))  # force applied to the ball
	conf = PinBallConfig(T, conf.start_pos, conf.target_pos, target_radius, ball_radius, noise, drag, force)


	ball = BallState(T, (conf.start_pos[1], conf.start_pos[2]), ball_radius)
	return PinBall(T,conf, deterministic_start, obstacles, ball)
end

function get_state_desc(env::PinBall{T})::Array{T,2} where {T<:Real}
    ranges = zeros(T, (4,2))
    ranges[1,:] .= [0., 1.]  # x range
	ranges[2,:] .= [0., 1.]  # y range
	ranges[3,:] .= [-2., 2.] # xdot range
	ranges[4,:] .= [-2., 2.] # ydot range
    return ranges
end

function get_action_desc(env::PinBall{T,TA})::Int where{T<:Real,TA<:Integer}
    return 5
end

function step!(env::PinBall, action::Int, rng::AbstractRNG)
    if action <= 0 || action > 5
        error("Action needs to be an integer in [1, 5]")
    end

	if rand(rng) < env.config.noise
		action = rand(rng, Int(1):Int(5))
	end
    # add action effect
    if action == 1
        add_impulse!(env.ball, env.config.force, 0.)  # Acc x
    elseif action ==2
        add_impulse!(env.ball, 0., -env.config.force) # Dec y
    elseif action ==3
        add_impulse!(env.ball, -env.config.force, 0.) # Dec x
    elseif action == 4
        add_impulse!(env.ball, 0., env.config.force) # Acc y
    else
        add_impulse!(env.ball, 0., 0.)  # No action
    end

	dxdy = [0., 0.]

    for i in 1:20
        stepball!(env.ball, env.dt)

        ncollision = 0
        fill!(dxdy, 0.)

        for obs in env.obstacles
            hit, double_collision, intercept = collision(obs, env.ball)
            if hit
				dxdy .= dxdy .+ collision_effect(env.ball, hit, double_collision, intercept)
                ncollision += 1
            end
		end
        if ncollision == 1
            env.ball.xDot = dxdy[1]
            env.ball.yDot = dxdy[2]
            if i == 19
                stepball!(env.ball, env.dt)
            end
        elseif ncollision > 1
            env.ball.xDot = -env.ball.xDot
            env.ball.yDot = -env.ball.yDot
        end
		env.t += env.dt
        found_goal = at_goal(env.ball, env.config)
		done = found_goal || env.t > 1000.

        env.done = done
        if done
            env.reward = (10000. * found_goal) - (1 - found_goal)
            update_state!(env)
            return env.reward
        end
    end

    add_drag!(env.ball, env.config.drag)
    checkbounds!(env.ball)

    if action == 5
        env.reward = -1.
    else
        env.reward = -5.
    end

    update_state!(env)

    return env.reward
end

function checkbounds!(ball::BallState)
    if ball.x > 1.0
        ball.x = 0.95
    end
    if ball.x < 0.0
        ball.x = 0.05
    end
    if ball.y > 1.0
        ball.y = 0.95
    end
    if ball.y < 0.0
        ball.y = 0.05
    end
end

function update_state!(env::PinBall)
    env.state .= env.ball.x, env.ball.y, env.ball.xDot, env.ball.yDot
end

function at_goal(ball::BallState, config::PinBallConfig)
	norm((ball.x, ball.y) .- config.target_pos) < config.target_radius
end

function terminalCheck(env::PinBall)
    return at_goal(env.ball, env.config) || env.t > 1000.
end

function reset_ball!(ball::BallState{T}, start_pos::Tuple{T,T}, deterministic_start::Bool, rng::AbstractRNG) where {T}
	ball.x = start_pos[1]
	ball.y = start_pos[2]
	if !deterministic_start
		ball.x += 0.02 * randn(rng, T)
		ball.y += 0.02 * randn(rng, T)
	end
	ball.xDot = 0.
	ball.yDot = 0.
end

function env_reset!(env::PinBall, rng::AbstractRNG)
	reset_ball!(env.ball, env.config.start_pos, env.deterministic_start, rng)
	update_state!(env)
	env.t = 0.
	env.done = false
end


function is_terminal(env::PinBall)::Bool
    return env.done
end


function clone(env::PinBall{T})::PinBall{T} where {T}
	env2 = PinBall(T, env.config, env.deterministic_start, env.obstacles, env.ball)
	env2.state .= env.state
	env2.reward = env.reward
	env2.done = env.done
	env2.dt = env.dt
	env2.t = env.t

	return env2
end

function add_impulse!(ball::BallState{T}, Δx::T, Δy::T) where {T}
    xDot = ball.xDot + Δx
    yDot = ball.yDot + Δy
    ball.xDot = clamp(xDot, -2., 2.)
    ball.yDot = clamp(yDot, -2., 2.)
end

function add_drag!(ball::BallState{T}, drag::T) where {T}
    ball.xDot *= drag
    ball.yDot *= drag
end

function stepball!(ball::BallState{T}, dt::T) where {T}
    ball.x += ball.xDot * ball.radius * dt
    ball.y += ball.yDot * ball.radius * dt
end


function read_config(::Type{T}, source) where {T}
	source = joinpath(@__DIR__, "configs", source)
    obstacles = Array{PinballObstacle{T}, 1}()
    target_pos = Tuple{T,T}((0.,0.))
    target_radius = T(0.04)
    ball_radius = T(0.02)
    start_pos = Tuple{T,T}((0.,0.))
    noise = T(0.)
    drag = T(0.995)
    force = T(1. / 5.)
    lines = readlines(source)
    for line in lines
        tokens = split(strip(line))
        if length(tokens) <= 0
            continue
        elseif tokens[1] == "polygon"
			nums = map(x->parse(T, x), tokens[2:end])
			points = [(x,y) for (x,y) in zip(nums[1:2:end], nums[2:2:end])]
            push!(obstacles, PinballObstacle(T, points))
        elseif tokens[1] == "target"
            target_pos = Tuple{T,T}(map(x->parse(T,x), tokens[2:3]))
            target_radius = parse(T,tokens[4])
        elseif tokens[1] == "start"
            start_pos = Tuple{T,T}(map(x->parse(T,x), tokens[2:3]))
        elseif tokens[1] == "ball"
            ball_radius = parse(T, tokens[2])
        end
    end

    conf = PinBallConfig(T, start_pos, target_pos, target_radius, ball_radius, noise, drag, force)

    return obstacles, conf
end

function collision(obs::PinballObstacle{T}, ball::BallState{T}) where {T}
    if ball.x - ball.radius > obs.maxx
        return false, nothing, nothing
	end
	if ball.x + ball.radius < obs.minx
        return false, nothing, nothing
	end
    if ball.y - ball.radius > obs.maxy
        return false, nothing, nothing
	end
    if ball.y + ball.radius < obs.miny
        return false, nothing, nothing
	end
    double_collision = false
    intercept_found = false

    i = 1
    j = 2
    intercept = nothing
    while i ≤ length(obs.points)
        p1, p2 = obs.points[i], obs.points[j]
        if intercept_edge(p1, p2, ball)
            if intercept_found
                intercept = select_edge((p1, p2), intercept, ball)
                double_collision = true
            else
                intercept = (p1, p2)
                intercept_found = true
            end
        end
        i += 1
        j += 1
        if j > length(obs.points)
            j = 1
        end
    end
    return intercept_found, double_collision, intercept
end

function collision_effect(ball::BallState{T}, intercept_found::Bool, double_collision::Bool, intercept::Tuple{Tuple{T,T},Tuple{T,T}})::Tuple{T,T} where {T}
    if double_collision
        return -ball.xDot, -ball.yDot
    end

    obstacle_vector = intercept[2] .- intercept[1]
    if obstacle_vector[1] < 0.
        obstacle_vector = intercept[1] .- intercept[2]
    end

    velocity_vector = (ball.xDot, ball.yDot)
    θ = compute_angle(velocity_vector, obstacle_vector) - π
    if θ < 0.
        θ += 2π
    end

    intercept_theta = compute_angle([-1, 0], obstacle_vector)
    θ += intercept_theta

    velocity = norm(velocity_vector)

    return velocity * cos(θ), velocity * sin(θ)
end

function compute_angle(v1, v2)
    angle_diff = atan(v1[1], v1[2]) - atan(v2[1], v2[2])
    if angle_diff < 0.
        angle_diff += 2π
    end
    return angle_diff
end


function intercept_edge(p1::Tuple{T,T}, p2::Tuple{T,T}, ball::BallState{T}) where {T}
    edge = p2 .- p1
    pball = (ball.x, ball.y)
    difference = pball .- p1

    scalar_proj = dot(difference, edge) / dot(edge, edge)
    scalar_proj = clamp(scalar_proj, 0., 1.)

    closest_pt = p1 .+ (edge .* scalar_proj)
    obstacle_to_ball = pball .- closest_pt
    distance = dot(obstacle_to_ball, obstacle_to_ball)

    if distance <= ball.radius^2
        # collision if the ball is not moving away
        velocity = (ball.xDot, ball.yDot)
        ball_to_obstacle = closest_pt .- pball

        angle = compute_angle(ball_to_obstacle, velocity)
        if angle > π
            angle = 2π - angle
        end

        if angle > (π / 1.99)
            return false
        end
        return true
    else
        return false
    end
end

function select_edge(intersect1::Tuple{Tuple{T,T},Tuple{T,T}}, intersect2::Tuple{Tuple{T,T},Tuple{T,T}}, ball::BallState{T}) where {T}
    velocity = (ball.xDot, ball.yDot)
    obstacle_vector1 = intersect1[2] .- intersect1[1]
    obstacle_vector2 = intersect2[2] .- intersect2[1]
    angle1 = compute_angle(velocity, obstacle_vector1)
    if angle1 > π
        angle1 -= π
    end

    angle2 = compute_angle(velocity, obstacle_vector2)
    if angle2 > π
        angle2 -= π
    end

    if abs(angle1 - π / 2.) < abs(angle2 - π / 2.)
        return intersect1
    else
        return intersect2
    end
end

# function circleShape(x, y, r)
# 	θ = LinRange(0, 2*π, 500)
# 	x .+ r*sin.(θ), y .+ r*cos.(θ)
# end

@recipe function f(env::PinBall)
	ballx, bally, bradius = env.ball.x, env.ball.y, env.ball.radius
	obstacles = env.obstacles
	tx, ty = env.config.target_pos
	tr = env.config.target_radius


	legend := false
	xlims := (0., 1.)
	ylims := (0., 1.)
	grid := false
	ticks := nothing
	foreground_color := :white
	aspect_ratio := 1.

	# obstacles
	for ob in obstacles
		@series begin
			seriestype := :shape
			seriescolor := :blue
			xpts = [p[1] for p in ob.points]
			ypts = [p[2] for p in ob.points]

			xpts, ypts
		end
	end

	# goal
	@series begin
		seriestype := :shape
		linecolor := nothing
		color := :green
		aspect_ratio := 1.
		fillalpha := 0.4

		circleShape(tx, ty, tr)
	end

	# ball
	@series begin
		seriestype := :shape
		linecolor := nothing
		color := :red
		aspect_ratio := 1.

		circleShape(ballx, bally, bradius)
	end

end
