module EvaluationOfRLAlgs

abstract type AbstractEnvironment end
export AbstractEnvironment
export get_state, step, reset, is_terminal, get_state_desc, get_action_desc, randomize_env, get_gamma

abstract type AbstractAgent end
abstract type AbstractPolicyAgent <: AbstractAgent end
abstract type AbstractValueAgent <: AbstractAgent end

export AbstractAgent, AbstractPolicyAgent
export new_episode!, act!

abstract type AbstractFuncApprox end
export AbstractFuncApprox
abstract type AbstractOptimizer <: AbstractFuncApprox end
abstract type AbstractTraceOptimizer <: AbstractOptimizer end
abstract type AbstractStepSizeOptimizer <: Any end
export AbstractOptimizer, AbstractTraceOptimizer, AbstractStepSizeOptimizer

export call, call!, gradient!, gradient, call_gradient, call_gradient!
export get_params, copy_params!, copy_params, get_num_params, set_params!, clone, get_num_outputs, get_output_range
export update!, add_to_params!



abstract type AbstractPolicy <: AbstractFuncApprox end
export AbstractPolicy
export get_action, get_action!, gradient_logp, gradient_logp!, get_action_gradient_logp!, gradient_entropy, logprob, entropy

abstract type AbstractCritic end
export AbstractCritic


abstract type AbstractModel end
export AbstractModel
export predict, predict!, sample

export run_agent!, run_agent, run_agent_parallel, eval_agent, eval_policy

using Random

include("functionapproximation/funcappx.jl")
include("functionapproximation/optimizers/optim.jl")
include("environments/environment.jl")
include("policies/policy.jl")
include("agents/agent.jl")
include("utils/utils.jl")

end
