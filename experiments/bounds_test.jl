using CSV
using Statistics
using StatsBase
using Random
using Distributions
using LinearAlgebra
using SparseArrays
using Bootstrap
using ArgParse
import Bootstrap:draw!
import Base:copy, length

using JLD

BLAS.set_num_threads(1)

function make_strats(num_strats)
    K = length(num_strats)
    strats = [1:n for n in num_strats]
    idx2strat = Dict{Int64, NTuple{K,Int64}}([n => s for (n, s) in enumerate(Iterators.product(strats...))]...)
    strat2idx = Dict{NTuple{K,Int64}, Int64}([s => n for (n, s) in enumerate(Iterators.product(strats...))]...)
    strats_tuples = [s  for s in Iterators.product(strats...)]
    alt_strats = []
    for k in 1:K
        palts = []
        for sidx in 1:length(strats_tuples)
            psalts = []
            s = idx2strat[sidx]
            for i in 1:num_strats[k]
                ns = ntuple(j->(j==k ? i : s[j]), length(s))
                push!(psalts, strat2idx[ns])
            end
            push!(palts, psalts)
        end
        push!(alt_strats, palts)
    end

    return strats_tuples, idx2strat, strat2idx, alt_strats
end

function compute_eta(num_strats)
    return 1.0 / sum(num_strats .- 1)
end

function makeCinfinite_dense!(C, strats, alts, players, payoffs, η, m)
    numS = length(strats)
    j=0
    f1 = 0.0
    f2 = 0.0
    tie = η / m
    for i in 1:numS
        for k in 1:players
            for j in alts[k][i]
                if j == i
                    continue
                else
                    f1 = payoffs(k, j)
                    f2 = payoffs(k, i)
                    if f1 > f2
                        C[i,j] = η
                    #elseif f2 > f1
                    #    C[i,j] = 0.0
                    elseif isapprox(f1,f2)
                        C[i,j] = tie
                    #else
                    #    throw(Error("unknown condition: f1: $(f1) f2: $(f2)"))
                    end
                end
            end
        end
    end
    for s in 1:numS
        C[s,s] = 1.0 - sum(view(C,s,:))
    end
    return C
end

function makeCinfinite_bounds(strats, alts, players, payoffs, η, m)
    Ic = Array{Int64,1}();
    Jc = Array{Int64,1}();
    Vl = Array{Float64,1}();
    Vu = Array{Float64,1}();
    num_uncertain = 0
    Eu = Array{Array{Int64,1},1}()
    numS = length(strats)
    tie = η/m
    for i in 1:numS
        uncertain = Array{Int64,1}()
        push!(uncertain, i)
        for k in 1:players
            for j in alts[k][i]
                if j == i
                    continue
                else
                    f1_low  = payoffs(k, j, :left)
                    f1_high = payoffs(k, j, :right)
                    f2_low  = payoffs(k, i, :left)
                    f2_high = payoffs(k, i, :right)
                    push!(Ic, i)
                    push!(Jc, j)
                    if (f1_low > f2_high)
                        push!(Vl, η)
                        push!(Vu, η)
                    elseif f2_low > f1_high
                        push!(Vl, 0)
                        push!(Vu, 0)
                    elseif isapprox(f1_low,f2_low) && isapprox(f1_high, f2_high)
                        push!(Vl, tie)
                        push!(Vu, tie)
                    else
                        push!(Vl, 0.0)
                        push!(Vu, η)
                        push!(uncertain, j)
                        num_uncertain += 1
                    end
                end
            end
        end
        push!(Eu, uncertain)
    end
    numS = length(strats)
    Clow = sparse(Ic, Jc, Vl, numS, numS)
    Chigh = sparse(Ic, Jc, Vu, numS, numS)
    dropzeros!(Clow)
    dropzeros!(Chigh)
    for s in 1:numS
        ch = 1.0 - sum(Clow[s,:])
        Clow[s,s] = 1.0 - sum(Chigh[s,:])
        Chigh[s,s] = ch
    end
    return Clow, Chigh, Eu, num_uncertain
end

function valueiteration_row4sp!(Pi, Pi_low, Pi_high, Ri, v, γ, numS, p0, rpert; Eu)
    Pi[Eu] .= Pi_low[Eu]
    if length(Eu) == 0
        return nothing
    end
    w = zeros(length(Eu))

    @. w =  (Ri + γ * v[Eu] + rpert[Eu])
    idxs = sortperm(w)
    p = p0

    for idx in idxs[end:-1:1]
        j = Eu[idx]
        pjh = Pi_high[j]
        pjl = Pi_low[j]
        dp = min(pjh-pjl, 1.0 - p)
        Pi[j] = pjl+dp
        p += dp
    end
end

vierror_bound(ϵ, γ) = (2*ϵ*γ)/(1-γ)

function valueiteration_contagg(P_low, P_high, Eu, Rs, numS, direction; γ=(1.0 - 1e-8))
    R = zeros(size(Rs))
    if direction == :max
        R .= Rs
    elseif direction == :min
        R .= -Rs
    else
        println("ERROR unrecognized symbol: ", direction)
        return NaN
    end
    v = zeros(numS)
    vold = zeros(numS)
    P = zeros((numS, numS))
    Ptmps = zeros((numS, numS))
    Ptmps .= P_low
    P .= P_low
    P0 = [sum(P_low[i, :]) for i in 1:numS]
    rper = rand(Float64, numS) .* (1e-10 / (1-γ))
    iteration = 0
    changed = true
    while changed
        t0 = time_ns()
        iteration += 1
        if iteration > 400
            ϵ = norm(v .- vold, Inf)
            bound = vierror_bound(ϵ, γ) # bound on max distance to optimal value function for any state
            bound = (1.0 - γ) * bound # bound on max error to bound on alpha rank (do this to converge regradless of gamma choice)
            println("hit iteration limit, epsilon = $(ϵ), bound = $(bound)")
            break
        end
        vold .= v
        changed = false
        for s in 1:numS
            edges = Eu[s]

            valueiteration_row4sp!(view(Ptmps, s, :), view(P_low, s, :), view(P_high, s, :), R[s], vold, γ, numS, P0[s], rper, Eu=edges)
            if any(abs.(view(Ptmps, s, edges) .- view(P, s, edges)) .>= 1e-8)
                changed=true
            end
            P[s, edges] .= view(Ptmps,s, edges)
        end
        t1 = time_ns()
        v .= inv(I - γ .* P) * R
        ϵ = norm(v .- vold, Inf)
        t2 = time_ns()
        tot = (t2-t1) / 1.0e9
        bound = vierror_bound(ϵ, γ) # bound on max distance to optimal value function for any state
        bound = (1.0 - γ) * bound # bound on max error to bound on alpha rank (do this to converge regradless of gamma choice)
        if bound < 1e-7
            changed = false
        end
        tf = time_ns()
        tot2 = (tf-t0) / 1.0e9
    end
    sol = (1.0 - γ) * mean(abs.(v))
    return sol
end


function (ecdf::ECDF)(x::Real, δ, tail=:both, method=:DKW)
    isnan(x) && return NaN
    p = ecdf(x)
    n = length(ecdf.sorted_values)
    ϵ = √(log(2.0/δ)/2n)
    if tail == :both
        return (max(p-ϵ, 0.0), min(p+ϵ, 1.0))
    elseif tail == :left
        return max(p-ϵ, 0.0)
    elseif tail == :right
        return min(p+ϵ, 1.0)
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end

function (ecdf::ECDF)(x::Real, δ, a::Real, b::Real, tail=:both, method=:DKW)
    #TODO make epsilon depend on tail, tail=both used delta/2
    isnan(x) && return NaN
    n = length(ecdf.sorted_values)
    ϵ = √(log(2.0/δ)/2n)
    pl = 0.0
    ph = 0.0
    if x < a
        pl = 0.0
        ph = 0.0
    elseif x ≥ b
        pl = 1.0
        ph = 1.0
    else
        p = ecdf(x)
        pl = max(p-ϵ, 0.0)
        ph = min(p+ϵ, 1.0)
    end


    if tail == :both
        return pl, ph
    elseif tail == :left
        return pl
    elseif tail == :right
        return ph
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end

function andersons_mean(D, δ, a, b, tail=:both)
    # assume D sorted
    # alpha is cvar threshold
    # δ is CI to hold with probability 1-δ
    # a is the lower bound on the Data
    # b is the upper bound on the Data
    N = length(D)
    ϵ = √(log(2.0/δ)/(2*N))
    if tail == :both
        Z = vcat(D, b)
        tmp = clamp.(collect(Float64, 1:N) ./ N .- ϵ, 0.0, Inf)
        upper = b - sum(diff(Z) .* tmp)

        Z = vcat(a, D)
        tmp = clamp.(collect(Float64, 0:N-1) ./ N .+ ϵ, -Inf, 1.0)
        lower = D[end] - sum(diff(Z) .* tmp)
        return lower, upper
    elseif tail == :left
        Z = vcat(a, D)
        tmp = clamp.(collect(Float64, 0:N-1) ./ N .+ ϵ, -Inf, 1.0)
        lower = D[end] - sum(diff(Z) .* tmp)
        return lower
    elseif tail == :right
        Z = vcat(D, b)
        tmp = clamp.(collect(Float64, 1:N) ./ N .- ϵ, 0.0, Inf)
        upper = b - sum(diff(Z) .* tmp)
        return upper
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end

function andersons_meangx(D, g, δ, a, b, tail=:both)
    # assume D sorted
    # alpha is cvar threshold
    # δ is CI to hold with probability 1-δ
    # a is the lower bound on the Data
    # b is the upper bound on the Data
    N = length(D)
    ϵ = √(log(2.0/δ)/(2*N))
    if tail == :both
        Z = vcat(D, b)
        gZ = g.(Z)
        tmp = clamp.(collect(Float64, 1:N) ./ N .- ϵ, 0.0, Inf)
        upper = gZ[end] - sum(diff(gZ) .* tmp)

        Z .= vcat(a, D)
        gZ .= g.(Z)
        tmp = clamp.(collect(Float64, 0:N-1) ./ N .+ ϵ, -Inf, 1.0)
        lower = gZ[end] - sum(diff(gZ) .* tmp)
        return lower, upper
    elseif tail == :left
        Z = vcat(a, D)
	gZ = g.(Z)
        tmp = clamp.(collect(Float64, 0:N-1) ./ N .+ ϵ, -Inf, 1.0)
        lower = gZ[end] - sum(diff(gZ) .* tmp)
        return lower
    elseif tail == :right
        Z = vcat(D, b)
        gZ = g.(Z)
        tmp = clamp.(collect(Float64, 1:N) ./ N .- ϵ, 0.0, Inf)
        upper = gZ[end] - sum(diff(gZ) .* tmp)
        return upper
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end

function cdfnormalized_mean(Fx, Fy, δx, δy, a, b, tail=:both)
    # Fx is normalizing distribution
    # Fy is empirical cdf of input variable
    # δx is CI to hold with probability 1-δx for Fx
    # δy is CI to hold with probability 1-δx for Fy
    # a is the lower bound on the distribution of X and Y
    # b is the upper bound on the distribution of X and Y
    y = Fy.sorted_values
    if tail == :both
        gl(x) = Fx(x, δx, a, b, :left, :DKW)
        lower = andersons_meangx(y, gl, δy, a, b, :left)
        gu(x) = Fx(x, δx, a, b, :right, :DKW)
        upper = andersons_meangx(y, gu, δy, a, b, :right)
        return lower, upper
    elseif tail == :left
        g(x) = Fx(x, δx, a, b, :left, :DKW)
        lower = andersons_meangx(y, g, δy, a, b, :left)
        return lower
    elseif tail == :right
        g(x) = Fx(x, δx, a, b, :right, :DKW)
        upper = andersons_meangx(y, g, δy, a, b, :right)
        return upper
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end



struct PerfDists
    dists::Dict{String,Dict{String,ECDF{Array{Float64,1}}}}
    algs::Array{String,1}
    envs::Array{String,1}
    bounds::Dict{String,Tuple{Float64,Float64}}
end

function copy(x::PerfDists)
    return deepcopy(x)
end

# function size(x::PerfDists)
#     return (length(x.algs), length(x.envs), length(x.dists[x.algs[1]][x.envs[1]].sorted_values))
# end

function length(x::PerfDists)
    tot = 0
    for alg in x.algs
        for env in x.envs
            tot += length(x.dists[alg][env].sorted_values)
        end
    end
    return tot
end

function draw!(x::PerfDists, o::PerfDists)
    algs = x.algs
    envs = x.envs
    for alg in algs
        for env in envs
            d = x.dists[alg][env].sorted_values
            idx = sample(1:length(d), length(d))
            o.dists[alg][env] = ecdf(d[idx])
        end
    end
end

function make_perfs_nobound(D::PerfDists)
    n,m = length(D.algs), length(D.envs)
    A = zeros(Float64, (n,m,n))
    N = length(D.dists[D.algs[1]][D.envs[1]].sorted_values)
    pts = zeros(N)
    for (i,alg) in enumerate(D.algs)
        for (j,env) in enumerate(D.envs)
            for (k,algnorm) in enumerate(D.algs)
                samedef = false
                if occursin("gw", env) || occursin("chain", env)
                    if (occursin("ac", alg) && occursin("ac", algnorm)) || (occursin("sarsa", alg) && occursin("sarsa", algnorm)) || (occursin("qlambda", alg) && occursin("qlambda", algnorm))
                        if (occursin("scaled", alg) && occursin("normal", algnorm)) || (occursin("normal", alg) && occursin("scaled", algnorm))
                            samedef = true
                        end
                    end
                end
                if i==k || samedef
                    A[i,j,k] = 0.5
                else
                    x = D.dists[alg][env].sorted_values
                    pts .= D.dists[algnorm][env].(x)
                    p = mean(pts)
                    A[i,j,k] = p
                end
            end
        end
    end
    return A
end

function make_perfs_bound(D::PerfDists, δ, bound=:DKW)
    n,m = length(D.algs), length(D.envs)
    δ′ = δ/(n*m)
    A = zeros(Float64, (n,m,n))
    Alow = zeros(Float64, (n,m,n))
    Ahigh = zeros(Float64, (n,m,n))
    for (i,alg) in enumerate(D.algs)
        for (j,env) in enumerate(D.envs)
            a,b = D.bounds[env]
            for (k,algnorm) in enumerate(D.algs)
                samedef = false
                if occursin("gw", env) || occursin("chain", env)
                    if (occursin("ac", alg) && occursin("ac", algnorm)) || (occursin("sarsa", alg) && occursin("sarsa", algnorm)) || (occursin("qlambda", alg) && occursin("qlambda", algnorm))
                        if (occursin("scaled", alg) && occursin("normal", algnorm)) || (occursin("normal", alg) && occursin("scaled", algnorm))
                            samedef = true
                        end
                    end
                end
                if i==k || samedef
                    A[i,j,k] = 0.5
                    Alow[i,j,k] = 0.5
                    Ahigh[i,j,k] = 0.5
                else
                    x = D.dists[alg][env].sorted_values
                    N = length(x)
                    pts = zeros(N)
                    pts .= D.dists[algnorm][env].(x)
                    p = mean(pts)
                    A[i,j,k] = p

                    if bound == :AndersonDKW
                        pts .= D.dists[algnorm][env].(x, δ′, a, b, :left, :DKW)
                        pl = andersons_mean(pts, δ′, 0, 1, :left)
                        Alow[i,j,k] = pl
                        pts .= D.dists[algnorm][env].(x, δ′, a, b, :right, :DKW)
                        pu = andersons_mean(pts, δ′, 0, 1, :right)
                        Ahigh[i,j,k] = pu
                    elseif bound == :DKW
                        pl, pu = cdfnormalized_mean(D.dists[algnorm][env], D.dists[alg][env], δ′, δ′, a, b, :both)
                        Alow[i,j,k] = pl
                        Ahigh[i,j,k] = pu
                    elseif bound == :TTest
                        δ2 = δ / (n*n*m)
                        tstar = quantile(TDist(N-1), 1 - δ2/2)
                        μ, σ = mean(pts), std(pts)
                        pl = μ - (σ / √N) * tstar
                        pu = μ + (σ / √N) * tstar
                        Alow[i,j,k] = pl
                        Ahigh[i,j,k] = pu
                    elseif bound == :TTest2
                        tstar = quantile(TDist(N-1), 1 -  δ′/2)
                        μ, σ = mean(pts), std(pts)
                        pl = μ - (σ / √N) * tstar
                        pu = μ + (σ / √N) * tstar
                        Alow[i,j,k] = pl
                        Ahigh[i,j,k] = pu
                    end
                end
            end
        end
    end
    return A, Alow, Ahigh
end

function load_results(paths, envs, algs)
    D = Dict{String,Dict{String,ECDF{Array{Float64,1}}}}()
    ab = Dict{String,Tuple{Float64,Float64}}()
    for env in envs
        ab[env] = (Inf, -Inf)
    end
    for alg in algs
        algD = Dict{String,ECDF{Array{Float64,1}}}()
        D[alg] = algD
    end
    for (path, env) in zip(paths, envs)
        df = CSV.read(path)
        for alg in algs
            D[alg][env] = ecdf(df[:, Symbol(alg)])
            a,b = D[alg][env].sorted_values[[1, end]]
            a = min(ab[env][1], a)
            b = max(ab[env][2], b)
            ab[env] = (a,b)
        end
    end
    perfs = PerfDists(D, algs, envs, ab)
    return perfs
end


function compute_aggregate(perfs::PerfDists, moves, strats, idx2strat, strat2idx, alts)
    A = make_perfs_nobound(perfs)
    numA = length(perfs.algs)
    numE = length(perfs.envs)
    aidx(i) = Int((i-1) ÷ numE)+1
    eidx(i) = Int((i-1) % numE)+1

    function payoffs(n,s::Int,tail=:none)
        if n == 1
            return A[s]
        else
            return -A[s]
        end
    end

    function payoffs(n,s::Tuple,tail=:none)
        m1, m2 = s
        j,k = eidx(m2), aidx(m2)

        if n == 1
            return A[m1,j,k]
        else
            return -A[m1,j,k]
        end
    end

    players = length(moves)

    η = compute_eta(moves);
    min_p = 0.0
    m = 50
    numS = length(strats)
    C = zeros((numS, numS))
    makeCinfinite_dense!(C, strats, alts, players, payoffs, η, m);
    R = zeros(numS)
    γ = (numS - 1.0) / Float64(numS)

    y = zeros(Float64, numA)
    invC = inv(I - γ .* C)
    for i in 1:numA
        for s in 1:numS
            R[s] = A[i,eidx(idx2strat[s][2]),aidx(idx2strat[s][2]),1]
        end
        v = invC * R
        y[i] = (1-γ) * mean(abs.(v))
    end
    return y
end

function compute_aggbound(perfs, moves, strats, idx2strat, strat2idx, alts, δ, bound=:DKW)
    A, Alow, Ahigh = make_perfs_bound(perfs, δ, bound)
    numA = length(perfs.algs)
    numE = length(perfs.envs)
    aidx(i) = Int((i-1) ÷ numE)+1
    eidx(i) = Int((i-1) % numE)+1

    function payoffs(n,s::Int,tail=:none)
        if tail==:none
            if n == 1
                return A[s]
            else
                return -A[s]
            end
        elseif tail==:left
            if n == 1
                return Alow[s]
            else
                return -Ahigh[s]
            end
        elseif tail==:right
            if n==1
                return Ahigh[s]
            else
                return -Alow[s]
            end
        elseif tail==:both
            if n ==1
                return Alow[s], Ahigh[s]
            else
                return -Ahigh[s], -Alow[s]
            end
        else
            throw(ArgumentError("tail=$(tail) is invalid"))
        end
    end

    players = length(moves)

    η = compute_eta(moves);
    min_p = 0.0
    m = 50
    numS = length(strats)
    C = zeros((numS, numS))
    fill!(C, 0.0)
    makeCinfinite_dense!(C, strats, alts, players, payoffs, η, m);

    R = zeros(numS)
    γ = (numS - 1.0) / Float64(numS)

    y = zeros(Float64, numA)
    ylow = zeros(Float64, numA)
    yhigh = zeros(Float64, numA)
    invC = inv(I - γ .* C)
    for i in 1:numA
        for s in 1:numS
            R[s] = A[i,eidx(idx2strat[s][2]),aidx(idx2strat[s][2])]
        end
        v = invC * R
        y[i] = (1-γ) * mean(abs.(v))
    end

    Clow, Chigh, Eu, num_uncertain = makeCinfinite_bounds(strats, alts, players, payoffs, η, m);

    for i in 1:numA
        for s in 1:numS
            R[s] = Alow[i,eidx(idx2strat[s][2]),aidx(idx2strat[s][2])]
        end
        ylow[i] = valueiteration_contagg(Clow, Chigh, Eu, R, numS, :min, γ=γ)
        for s in 1:numS
            R[s] = Ahigh[i,eidx(idx2strat[s][2]),aidx(idx2strat[s][2])]
        end
        yhigh[i] = valueiteration_contagg(Clow, Chigh, Eu, R, numS, :max, γ=γ)
    end
    return hcat(y, ylow, yhigh)
end

function bootstrap_aggbound(perfs, moves, strats, idx2strat, strat2idx, alts, δ, num_boot)
    bs = bootstrap((x)->(compute_aggregate(x, moves, strats, idx2strat, strat2idx, alts)),
                    perfs, BasicSampling(num_boot)
    )
    numA = length(perfs.algs)
    numE = length(perfs.envs)
    cfbs = confint(bs, PercentileConfInt(1 - (δ/(numA*numE))))
    Y = hcat([cfbs[i][1] for i in 1:numA], [cfbs[i][2] for i in 1:numA], [cfbs[i][3] for i in 1:numA])
    return Y
end


function load_env_result(base_dir, env, alg, colnames, perf_col)
    path = join([base_dir, env, string("allres_", alg, ".csv")], "/")
    if occursin("gw", env) || occursin("chain", env)
        header = colnames[4:end]
    else
        header = colnames
    end
    f = CSV.File(path, header=header)
    N = length(f)
    data = zeros(N)
    for (i, row) in enumerate(f)
        data[i] = row[perf_col]
    end
    return data
end

function combine_algs(base_dir, env, algs, acols, perf_col)
    allalg = [load_env_result(base_dir, env, alg, cols, perf_col) for (alg, cols) in zip(algs, acols)]
    return allalg
end

function split_data(data, nsplits, sizes, save_dir, env, algs)
    min_samples = sum(nsplits .* sizes)
    println(min_samples, " ", length.(data))
    @assert all(length.(data) .>= min_samples)
    foreach(shuffle!, data)
    iters = ones(Int, length(algs))

    for N in sizes
        for i in 1:nsplits
            df = DataFrame()
            for (j, alg) in enumerate(algs)
                idx = iters[j]
                df[!, Symbol(alg)] = data[j][idx:idx+(N-1)]
                iters[j] = idx + N
            end
            path = join([save_dir,string("allalgs_", env, "_$(lpad(N, 5, '0'))_$(lpad(i, 5, '0')).csv")], "/")
            CSV.write(path, df)
        end
    end
end

function load_and_split(base_dir, save_dir, env, algs, acols, perf_col, nsplits, sizes)
    data = combine_algs(base_dir, env, algs, acols, perf_col)
    split_data(data, nsplits, sizes, save_dir, env, algs)
end

function make_paths(base_dir, envs, N, split_idx)
    paths = []
    for env in envs
        path = join([base_dir, string("allalgs_", env, "_$(lpad(N, 5, '0'))_$(lpad(split_idx, 5, '0')).csv")], "/")
        push!(paths, path)
    end
    return paths
end

function get_aggbounds_samples(base_dir, envs, algs, δ, bound, sample_sizes)
    numA = length(algs)
    numE = length(envs)
    moves = [numA, numA*numE]
    strats, idx2strat, strat2idx, alts = make_strats(moves);
    results = []
    num_splits = 1000
    for ss in sample_sizes
        sres = zeros((num_splits, numA, 3))
        for i in 1:num_splits #Threads.@threads for i = 1:num_splits
            paths = make_paths(base_dir, envs, ss, i)
            perfs = load_results(paths, envs, algs)
            sres[i, :, :] .= compute_aggbound(perfs, moves, strats, idx2strat, strat2idx, alts, δ, bound)
        end
        push!(results, sres)
    end
    return results
end

function bootstrap_aggbounds_samples(base_dir, envs, algs, δ, num_boot, sample_sizes)
    numA = length(algs)
    numE = length(envs)
    moves = [numA, numA*numE]
    strats, idx2strat, strat2idx, alts = make_strats(moves);
    results = []
    num_splits = 1000
    for ss in sample_sizes
        sres = zeros((num_splits, numA, 3))
        Threads.@threads for i = 1:num_splits
            paths = make_paths(base_dir, envs, ss, i)
            perfs = load_results(paths, envs, algs)
            sres[i, :, :] .= bootstrap_aggbound(perfs, moves, strats, idx2strat, strat2idx, alts, δ, num_boot)
        end
        push!(results, sres)
    end
    return results
end

function num_bound_violations(res)
    N = size(res)[1] # num trials
    violations = 0
    for i in 1:N
        perf = res[i, :, 1]
        for j in 1:N
            if i==j
                continue
            end
            below = perf .< res[j,:,2]
            above = perf .> res[j,:,3]
            if any(above) || any(below)
                violations += 1
            end
        end
    end
    return violations / (N * (N-1))
end

function num_significant(res)
    N, numA = size(res)[1:2] # num trials, num algorithms
    num_sig = 0
    for i in 1:N
        for j in 1:(numA-1)
            for k in j:numA
                jlow = res[i, j, 2]
                jhigh = res[i, j, 3]
                klow = res[i, k, 2]
                khigh = res[i, k, 3]
                num_sig += (jlow > khigh) | (jhigh < klow)
            end
        end
    end
    numComp = (numA * (numA-1)) / 2
    return num_sig / (N * numComp)
end

function bound_res(res)
    violations = zeros(size(res)[1])
    sigs = zeros(size(res)[1])
    for i in 1:length(violations)
        violations[i] = num_bound_violations(res[i])
        sigs[i] = num_significant(res[i])
    end
    return violations, sigs
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--conf", "-c"
            help = "confidence level"
            arg_type = Float64
            required = true
        "--dir", "-d"
            help = "directory"
            arg_type = String
            required = true
        "--sdir", "-s"
            help = "folder directory to store results"
            arg_type = String
            required = true
        "--boot", "-b"
            help = "number of bootstrap samples to use"
            arg_type = Int
            required = false
            default = 1000
    end

    parsed_args = parse_args(ARGS, s)
    base_dir = parsed_args["dir"]
    save_dir = parsed_args["sdir"]
    δ = parsed_args["conf"]
    num_boot = parsed_args["boot"]
    # δ = 0.05
    # num_boot = 10000
    # base_dir = "/home/sjordan/Documents/icmleval/complete/bounds/splits";
    # save_dir = "/home/sjordan/Documents/icmleval/complete/bounds/intervals2";
    println("data dir: ", base_dir)
    println("save dir: ", save_dir)
    println("confidence level: ", δ)
    println("num_bootstraps: ", num_boot)

    algs_bt = ["ac-parl2", "sarsa-scaled", "sarsa-parl2", "q-parl2"]
    envs_bt = ["mntcar_d", "chain10d", "chain10s", "chain50d", "chain50s", "gw10d", "gw10s", "gw5d", "gw5s", ]

    acp_cols = ["dorder", "iorder", "full", "gamma", "lam", "palpha", "logp", "life", "end"]
    qlp_cols = ["dorder", "iorder", "full", "gamma", "lam", "eps", "logp", "life", "end"]
    sls_cols = ["dorder", "iorder", "full", "gamma", "lam", "eps", "qalpha", "logp", "life", "end"]
    slp_cols = ["dorder", "iorder", "full", "gamma", "lam", "eps", "logp", "life", "end"]
    alg_cols_bt = [acp_cols, sls_cols, slp_cols, qlp_cols];
    sample_sizes = [10, 30, 100, 1000, 10000]

    DKW_res = get_aggbounds_samples(base_dir, envs_bt, algs_bt, δ, :DKW, sample_sizes);
    path = join([save_dir, "dkw3.jld"], "/")
    save(path, "DKW_res", DKW_res)
    # @save path DKW_res
    #@load path DKW_res
    dkw_bv, dkw_sig = bound_res(DKW_res)
    println(dkw_bv)
    println(dkw_sig)
    println("got DKW bounds")
    flush(stdout)

    TT_res = get_aggbounds_samples(base_dir, envs_bt, algs_bt, δ, :TTest, sample_sizes);
    path = join([save_dir, "tt3-AM.jld"], "/")
    save(path, "TT_res", TT_res)
    # @save path TT_res
    # @load path TT_res
    tt_bv, tt_sig = bound_res(TT_res)
    println(tt_bv)
    println(tt_sig)
    println("got TTest bounds")
    flush(stdout)

    TT_res2 = get_aggbounds_samples(base_dir, envs_bt, algs_bt, δ, :TTest2, sample_sizes);
    path = join([save_dir, "tt3-AAM.jld"], "/")
    save(path, "TT_res2", TT_res2)
    # @save path TT_res2
    # @load path TT_res2
    tt2_bv, tt2_sig = bound_res(TT_res2)
    println(tt2_bv)
    println(tt2_sig)
    println("got TTest2 bounds")
    flush(stdout)

    #@time BS_res = bootstrap_aggbounds_samples(base_dir, envs_bt, algs_bt, δ, num_boot, sample_sizes);
    #path = join([save_dir, "bs.jld"], "/")
    #save(path, "BS_res", BS_res)
    #@save path BS_res
    # @load path BS_res

    println("got Bootstrap bounds")
    flush(stdout)

    # dkw_bv, dkw_sig = bound_res(DKW_res)
    # tt_bv, tt_sig = bound_res(TT_res)
    # tt2_bv, tt2_sig = bound_res(TT_res2)
    #bs_bv, bs_sig = bound_res(BS_res);
    println("got all result measures")
    flush(stdout)

    bvdf = DataFrame(SampleSize = sample_sizes,
        vDKW=dkw_bv, vTTest=tt_bv, vTTest2=tt2_bv,
        #vBootstrap=bs_bv,
        sDKW=dkw_sig, sTTest=tt_sig, sTTest2=tt2_sig,
        # sBootstrap=bs_sig
                )
    println("Bound results")
    println(bvdf)

end

main()
