import EvaluationOfRLAlgs

function create_env(config, rng)
    name = config["name"]
    randomize = config["randomize"]

    atype = config["action_type"]
    if atype == "discrete"
        TA = Int
    elseif atype == "continuous"
        TA = Float64
    else
        println("action type ", atype, "not supported")
    end

    if name == "cartpole"
        if randomize
            env = EvaluationOfRLAlgs.randomize_env(EvaluationOfRLAlgs.CartPole{Float64,TA}, rng)
        else
            env = EvaluationOfRLAlgs.CartPole(Float64, TA)
        end
    elseif name == "mountaincar"
        start = config["stochasticstart"]
        if randomize
            env = EvaluationOfRLAlgs.randomize_env(EvaluationOfRLAlgs.MountainCar{Float64,TA}, rng, start)
        else
            env = EvaluationOfRLAlgs.MountainCar(Float64, TA, start)
        end
    elseif name == "acrobot"
        start = config["stochasticstart"]
        if randomize
            env = EvaluationOfRLAlgs.randomize_env(EvaluationOfRLAlgs.Acrobot{Float64,TA}, rng, start)
        else
            env = EvaluationOfRLAlgs.Acrobot(Float64, TA, start)
        end
    elseif name == "pendulum"
        start = config["stochasticstart"]
        if randomize
            env = EvaluationOfRLAlgs.randomize_env(EvaluationOfRLAlgs.Pendulum{Float64,TA}, rng, start)
        else
            env = EvaluationOfRLAlgs.Pendulum(Float64, TA, start)
        end
    elseif name == "pinball"
        start = config["stochasticstart"]
        mname = config["mapname"]
        if randomize
            env = EvaluationOfRLAlgs.randomize_env(EvaluationOfRLAlgs.PinBall{Float64}, rng, mname, start)
        else
            env = EvaluationOfRLAlgs.PinBall(Float64, mname, start)
        end
    elseif name == "bicycle"
        if randomize
            env = EvaluationOfRLAlgs.randomize_env(EvaluationOfRLAlgs.Bicycle{Float64}, rng)
        else
            env = EvaluationOfRLAlgs.Bicycle(Float64)
        end
    elseif name == "chain"
        transition = config["stransition"]
        size = config["size"]
        if randomize
            env = EvaluationOfRLAlgs.randomize_env(EvaluationOfRLAlgs.Chain{Int,Int}, rng, size, transition)
        else
            env = EvaluationOfRLAlgs.Chain(size, transition)
        end
    elseif name == "gridworld"
        transition = config["stransition"]
        size = config["size"]
        if randomize
            env = EvaluationOfRLAlgs.randomize_env(EvaluationOfRLAlgs.GridWorld{Int,Int}, rng, size, transition)
        else
            env = EvaluationOfRLAlgs.GridWorld(size, transition)
        end
    else
        println("environment ", name, " not supported yet")
    end


    return env
end
