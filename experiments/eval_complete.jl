using JSON, Random, ArgParse
import EvaluationOfRLAlgs

include("algconstructors.jl")
include("envconstructors.jl")

function runsweep(id, seed, envconfig, algname, save_dir, trials, paramdir=nothing)
    num_episodes = envconfig["num_episodes"]
    num_endepisodes = envconfig["num_endepisodes"]

    rng = Random.MersenneTwister(seed)

    save_name = "$(algname)_$(lpad(seed, 5, '0')).csv"

    save_path = joinpath(save_dir, save_name)

    open(save_path, "w") do f
        for trial in 1:trials
            env = create_env(envconfig, rng)
            agent, hyps, logp = sample_alg(algname, env, rng, num_episodes)
            meanret = 0.
            meaneret = 0.


            returns = EvaluationOfRLAlgs.run_agent!(env, agent, num_episodes, rng)
            meanret += mean(returns)
            ereturns = EvaluationOfRLAlgs.eval_agent(env, agent, num_endepisodes, rng)
            meaneret += mean(ereturns)

            result = join([hyps..., logp, meanret, meaneret], ',')
            write(f, "$(result)\n")
            flush(f)
            println("$trial \t $(result)")
            flush(stdout)
        end
    end

end



function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--alg-name", "-a"
            help = "name of the algorithm to run"
            arg_type = String
            required = true
        "--env-file", "-e"
            help = "file path to JSON file containing environment to run"
            arg_type = String
            required = true
        "--log-dir", "-l"
            help = "folder directory to store results"
            arg_type = String
            required = true
        "--id"
            help = "identifier for this experiment. Used in defining a random seed"
            arg_type = Int
            required = true
        "--seed", "-s"
            help = "random seed is seed + id"
            arg_type = Int
            required = true
        "--trials", "-t"
            help = "number of random trials to run"
            arg_type = Int
            default = 1
    end

    parsed_args = parse_args(ARGS, s)
    aname = parsed_args["alg-name"]
    efile = parsed_args["env-file"]
    println("Excuting algorithm: $aname")
    println("Loading environment config: $efile")
    println("Logging Directory: $(parsed_args["log-dir"])")
    flush(stdout)
    envconfig = JSON.parsefile(efile, dicttype=Dict, inttype=Int64)["env"]
    println("Environment config:")
    println(envconfig)
    flush(stdout)
    save_dir = parsed_args["log-dir"]
    save_dir = joinpath(save_dir, envconfig["logname"])
    mkpath(save_dir)
    println("Saving results in: $save_dir")
    trials = parsed_args["trials"]
    id = parsed_args["id"]
    seed = parsed_args["seed"]

    println("id $id seed=(id+seed) $(id + seed)")
    flush(stdout)
    seed = id + seed

    runsweep(id, seed, envconfig, aname, save_dir, trials)

end

main()
