using Statistics
using Random
using Distributions
import Future
import EvaluationOfRLAlgs

using Plots
gr()

# sample uniformly on natural log scale
function logRand(low, high, rng)
    X = exp(rand(rng, Uniform(log(low), log(high))))
    logp = -log(log(high) - log(high) - log(X))
    return X, logp
end

function quickeval()
    num_sweeps = 25
    results = zeros((num_sweeps,2))
    rng = Random.MersenneTwister(0)  # change 0 to change random seed

	# use this if you have set Blas to single threaded
    # rngs = [rng; accumulate(Future.randjump, fill(big(10)^20, num_sweeps-1), init=rng)]  # use different random number generator for each trial
	# Threads.@threads for s in 1:num_sweeps
    #     results[s, :] .= run_alg(rngs[s])
    # end

	# use this to get results sequentially
	for s in 1:num_sweeps
		results[s, :] .= run_alg(rng)
	end

	# get results for empirical quantile plots
    lrets = sort(results[:, 1])
	frets = sort(results[:, 2])
	xpts = collect(Float64, 1:num_sweeps) / num_sweeps
	# plot empirical quantile plot for both distributions
	p = plot(xpts, lrets, label="Average Return", color=:dodgerblue)
	p = plot!(p, xpts, frets, label="Final Return", color=:crimson)
	ylabel!(p, "Return")
	xlabel!(p, "Probability")
	display(p)
end


function run_alg(rng)
	env = MountainCar(Float64, Int, false) # takes about 20 seconds
	# env = PinBall(Float64, "pinball_medium.cfg") # takes about 2 minutes
    num_trials = 1
    num_episodes = 100

    state_ranges = get_state_desc(env)
    num_actions = get_action_desc(env)
    ϕ = FourierBasis(state_ranges, 5,6,false)  		# create a fourier basis function that use both sine and cosine functions with dependent order of 6 and independent order of 7
    p = LinearFunction(Float64, ϕ, num_actions)  	# create a linear function using the fourier basis to use as the policy
    policy = LinearSoftmaxPolicy(p)  				# make the policy a softmax of the function p
    vf = LinearFunction(Float64, ϕ, 1)  			# create a linear function for the value function
	qf = EvaluationOfRLAlgs.LinearFunction(Float64, ϕ, num_actions)  # creates a q function for use in Sarsa and Q learning algorithms
    λ = rand(rng, Uniform(0., 1.0))  				# sample a random elgibility trace decay
    γ = get_gamma(env)  							# get discount factor from environment (most are 1.0 so setting this lower could be useful)
	ϵ = rand(rng, Uniform(0.0, 0.1))  				# sample the #ϵ-greedy exploration parameter
	αp = logRand(1e-3, 1.0, rng)[1] / get_num_params(policy) # scale the learning rate by the number of parameters. Helps for setting linear rates of linear functions.
	#αv = logRand(1e-3, 1.0, rng)[1] / get_num_params(vf)  # learning rate for value functions when not using Parl2 (i.e., REINFORCE)

	# create the optimization functions
    popt = AccumulatingTraceOptimizer(policy, αp, γ, λ)
    vopt = Parl2AccumulatingTraceOptimizer(vf, γ, λ)
	# qopt = Parl2AccumulatingTraceOptimizer(qf, γ, λ)

	# create the algorithm
    agent = ActorCritic(policy, vf, popt, vopt, γ)
    # agent = NACTD(policy, vf, wf, popt, vopt, wopt, γ, true)  # exercise: create linear functon wf using IdentityBasis
    # agent = REINFORCE(policy, vf, αp, αv, γ, false, true)
    # agent = Sarsa(qf, qopt, γ, ϵ)
    # agent = QLearning(qf, qopt, γ, ϵ)

	# finite difference based search methods (black box optimization)
    # agent = BasicRandomSearch(policy, 0.01, 1.0, 1.0, 3, rng, num_episodes-10)
    # agent = AugmentedRandomSearch(policy, 0.015, 1.0, 0.5, 10, 5, rng, num_episodes-10)


    # returns = run_agent_parallel(env, agent, num_trials, num_episodes, rng)  # run multiple trials using several threads
    # returns = run_agent(env, agent, num_trials, num_episodes, rng) # run multiple trials using one thread
    learning_returns = run_agent!(env, agent, num_episodes, rng) # run one trial. the ! denotes that agent will be update. The two functions above do not update the agent.
	# returns = run_agent!(env, agent, num_episodes, rng, true)  # the true flag displays the agent during leraning. PyPlot backend is slow so use gr(). Do not run this in parallel or if you want speed.
	final_returns = eval_agent(env, agent, 30, rng) # evaluate updated agent after learning

	return mean(learning_returns), mean(final_returns)

end
