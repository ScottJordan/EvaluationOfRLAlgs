# EvaluationOfRLAlgs

This repository contains the code need to replicate the experiments in the paper:

Scott M. Jordan, Yash Chandak Daniel Cohen, Mengxue Zhang, Philip S. Thomas, **Evaluating the Performance of Reinforcement Learning Algorithms** (ICML, 2020) [[ICML](https://icml.cc/Conferences/2020/AcceptedPapersInitial)] [[ArXiv (not-yet-uploaded)](tbd)] [[Video (not-yet-uploaded)](tbd)] [[Code](https://github.com/scottjordan/EvaluationOfRLAlgs)]

## Citing

If you use this code please cite the paper using the following bib entry.

```
@inproceedings{jordan2020eval,
  title={Evaluating the {P}erformance of {R}einforcement {L}earning {A}lgorithms},
  author={Jordan, Scott M. and Chandak, Yash, Cohen, Daniel and Zhang, Mengxue and Thomas, Phillip S.},
  booktitle={Proceedings of the 37rd International Conference on International Conference on Machine Learning},
  year={2020}
}
```

## Installing the repository and dependicies
The recommended way to install the package is to clone this repository and add it to the Julia environment.

It can be added to the a Julia environment using the following command
```julia
] develop /path/to/EvaluationOfRLAlgs
```

Python is used for generating plots and has a few dependencies. The following line will add all necessary dependencies to an Anaconda environment.
```bash
conda install matplotlib jupyter pandas seaborn
```


## Code Structure
There are two main components of this code base:

* Code to run experiments from paper is in the [experiments/](experiments/) directory
* Code for the algorithms and environments is in the [src/](src/) directory

## Experiments

Current performance of algorithms:

<img src="resources/aggregate_perf.png" width="600">

### Generating new results

One can generate samples of performance for an algorithm using the following command:

```bash
julia -O 3 --math-mode ieee experiments/eval_complete.jl -a <alg_name> -e <environment_config_file> -l <results_directory>  --id <job_id> --seed <Int> --trials <Int>
```
`-a` The name of the algorithm to run, e.g., `-a sarsa-parl2`. See [experiments/algconstructors.jl](algconstructors.jl) for a list of all names.  
`-e` Path to a JSON containing the information about the environment to run. See [experiments/envconf/] for examples.  
`-l` Path to a place to store the results, e.g., `-l experiments/data/samples`.  
`--id` The job id parameter, useful for running on clusters.  
`--seed` An integer specifying the random seed to use. The random seed used for the experiment is `id+seed`.  
`--trials` is an integer specifying the number of trials to run for the algorithm, this should be greater than 100 and probably in the range of 1,000 to 10,000 to ensure statistically significant results.  

The results generated by this function is a CSV file containing for each trial: the hyperparameters used, the average return over the trial, and the average return using the final policy found (number of episodes is defined in environment config). These results will be printed during execution.

To run `Sarsa-Parl2` on `Cart-Pole` for 1,000 trials, execute the following command (from the experiments directory):
```bash
julia -O 3 --math-mode ieee eval_complete.jl -a sarsa-parl2 -e envconf/cp_d.json -l data/samples  --id 0 --seed 1234 --trials 1000
```

Note: Using an old laptop, discrete environments should run 1,000 trials in less than two minutes and Parl2 on Cart-Pole can take less than four hours for 1,000 trails  (better algorithms have longer running times on Cart-Pole). Running experiments in parallel is encourage to reduce experiment time. To run in parallel simply vary the random `seed` and/or the `id` parameter to make experiments independent of each other. It is also useful to tell Julia and BLAS libraries the number of threads to use for each process, e.g., the following command limits the process to two threads:

```bash
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export JULIA_NUM_THREADS=2
export JULIA_CPU_THREADS=2
```

### Getting data from the paper

The zip file [experiments/data/samples_from_paper.zip](samples_from_paper.zip) contains the results reported in the main paper and appendix. To use this data, unzip the archive in `experiments/data/` and change the folder name from `samples_from_paper` to `samples`. This data can then be directly used to reproduce the plots from the paper. The file `cat_results.sh` can be used to aggregate parallel runs of algorithms into a single file for analysis.

### Computing Aggregate Performance

The jupyter notebook [experiments/AggregatePerformance.ipynb](AggregatePerformance.ipynb) contains the code necessary to import the data, compute the aggregate performance with confidence intervals and save the results into a CSV for plotting. One should be able to just run all the cells of the notebook to generate the aggregate results. Note that the Percentile Bootstrap method can take four hours to run on the provided data.

### Reporting Results (Plots and Tables)

The jupyter notebook [experiments/eval_plots.ipynb](eval_plots.ipynb) contains code to generate plots and tables of the performance results. The code in the notebook can generate bar plots with confidence intervals for the aggregate performance, distribution plots with confidence intervals for the performance on each environment, and LaTeX code for showing tables of the performance, rankings, and uncertainty on each environment. There are also other plots that can be generated to match the ones produced in the paper.

### Adding your favorite algorithm
To add your favorite algorithm and run it on the provide environments only two things are required. First is a definition of the algorithm's type along with functions implementing the algorithm interface. See the implementations of existing algorithms for details, e.g., [src/agents/actorcritic.jl](actorcritic.jl).

The second step is to create a complete algorithm definition and an entry in [experiments/algconstructors.jl](algconstructors.jl). A complete algorithm definition specifies all hyperparameters, e.g., policy structure, basis function, step-size, given only meta-information about the environment, e.g., number of state features, size of action space. This makes it easy for both you and others to run the algorithm and evaluate its performance. Note that it is also acceptable if an algorithm does not perform well because it is difficult automatically set its hyperparameters. Poor performance often means that the information needed to make the agorithm perform well is unknown and further research is needed to make it useful. Evaluating an algorithm this way makes it clear when new knowledge makes an algorithm easier to use and successful. Just do not tune the algorithm's definition to this (or any) suite of environments as this will bias the performance results.

## Source Code
The source code in the repository is broken down into several main components, agents (algorithms), environments, function approximators (basis functions, linear, etc), policies, optimizers (Adam, eligibility traces, etc).

The following defines simple actor-critic with eligibility traces algorithm using Parl2 and runs it on a PinBall environment.

```julia
using Statistics
using Random
using Distributions
import EvaluationOfRLAlgs


function run_alg()
  rng = Random.MersenneTwister(0)  # change 0 to change seed
  env = EvaluationOfRLAlgs.PinBall(Float64, "pinball_medium.cfg")
  num_trials = 1
  num_episodes = 200

  state_ranges = get_state_desc(env)
  num_actions = get_action_desc(env)
  ϕ = FourierBasis(state_ranges, 6,7,true)  # create a Fourier basis function that use both sine and cosine functions with dependent order of 6 and independent order of 7
  p = LinearFunction(Float64, ϕ, num_actions)  # create a linear function using the Fourier basis to use as the policy
  policy = LinearSoftmaxPolicy(p)  # make the policy a softmax of the function p
  vf = LinearFunction(Float64, ϕ, 1)  # create a linear function for the value function
  λ = rand(rng, Uniform(0., 1.0))  # sample a random eligibility trace decay
  γ = get_gamma(env)  # get discount factor from environment (most are 1.0 so setting this lower could be useful)
  αp = 0.1 / get_num_params(policy) # scale the learning rate by the number of parameters. Helps for setting linear rates of linear functions.

  # create the optimization functions
  popt = AccumulatingTraceOptimizer(policy, αp, γ, λ)
  vopt = Parl2AccumulatingTraceOptimizer(vf, γ, λ)

	# create the algorithm
  agent = ActorCritic(policy, vf, popt, vopt, γ)

  learning_returns = run_agent!(env, agent, num_episodes, rng) # run one trial. the ! denotes that agent will be update. The two functions above do not update the agent.
  final_returns = eval_agent(env, agent, 30, rng) # evaluate updated agent after learning

  return mean(learning_returns), mean(final_returns)
end

run_alg()
```

This example and other variations are provide in the file [experiments/demo.jl](demo.jl) along with a simple way to display the distribution of performance for an algorithm.

## Contributing
If you would like to add an algorithm, environment, or new complete definition to the repository please create a pull request. We will leave version 1.0 as the version of the code that was used to produce the results in the paper.
