# Tabular Function file

using LinearAlgebra

struct LinearFunction{TP, TB} <: AbstractFuncApprox where {TP<:Real,TB<:AbstractFuncApprox}
    θ::Array{TP, 2}
    ϕ::TB
    feats::Array{TP, 1}
    LinearFunction(input_dim::Int, output_dim::Int) = new{Float64, IdentityBasis}(zeros(Float64,input_dim, output_dim), IdentityBasis(input_dim), zeros(Float64, 1))
    LinearFunction(::Type{T}, input_dim::Int, output_dim::Int) where {T<:Real} = new{T, IdentityBasis}(zeros(T,input_dim, output_dim), IdentityBasis(input_dim), zeros(T, 1))
    LinearFunction(::Type{T}, basis::AbstractFuncApprox, output_dim::Int) where {T<:Real} = new{T, typeof(basis)}(zeros(T,get_num_outputs(basis), output_dim), clone(basis), zeros(T, get_num_outputs(basis)))
    LinearFunction(θ::Array{T,2}, basis::AbstractFuncApprox, feats::Array{T,1}) where {T<:Real} = new{T, typeof(basis)}(θ, basis, feats)
end

function call(fun::LinearFunction{TP,IdentityBasis}, x::Int)::Array{TP,1} where {TP<:Real}
    return fun.θ[x, :]
end

function call!(out::Array{TP, 1}, fun::LinearFunction{TP,IdentityBasis}, x::Int) where {TP<:Real}
    out .= fun.θ[x, :]
end

function call(fun::LinearFunction{TP,IdentityBasis}, x::Int, y::Int)::TP where{TP<:Real}
    return fun.θ[x, y]
end

function call(fun::LinearFunction{TP,IdentityBasis}, x::Array{TP,1})::Array{TP,1} where {TP<:Real}
    return fun.θ'*x
end

function call!(out::Array{TP, 1}, fun::LinearFunction{TP,IdentityBasis}, x::Array{TP,1}) where {TP<:Real}
    # out .= fun.θ'*x
    for i in 1:size(fun.θ)[2]
        out[i] .= dot(view(fun.θ, :, i), x)
    end
end

function call(fun::LinearFunction{TP}, x::Array{TP,1})::Array{TP,1} where {TP<:Real}
    call!(fun.feats, fun.ϕ, x)
    return fun.θ'*fun.feats
end

function call!(out::Array{TP, 1}, fun::LinearFunction{TP}, x::Array{TP,1}) where {TP<:Real}
    call!(fun.feats, fun.ϕ, x) #
    # out .= fun.θ'*fun.feats
    for i in 1:size(fun.θ)[2]
        out[i] = dot(view(fun.θ, :, i), fun.feats)
    end
end

function call(fun::LinearFunction{TP,IdentityBasis}, x::Array{TP,1}, y::Int)::TP where {TP<:Real}
    return dot(view(fun.θ,:, y),x)
end

function call(fun::LinearFunction{TP}, x::Array{TP,1}, y::Int) where {TP<:Real}
    call!(fun.feats, fun.ϕ, x)
    return dot(view(fun.θ,:, y), fun.feats)
end


function gradient!(grad::Array{TP}, fun::LinearFunction{TP, IdentityBasis}, x::Int) where {TP<:Real}
    grad = reshape(grad, size(fun.θ))
    fill!(grad, 0.)
    grad[x, :] .= 1.
end

function gradient!(grad::Array{TP}, fun::LinearFunction{TP, IdentityBasis}, x::Int, y::Int) where {TP<:Real}
    grad = reshape(grad, size(fun.θ))
    fill!(grad, 0.)
    grad[x, y] = 1.
end

function gradient!(grad::Array{TP}, fun::LinearFunction{TP, IdentityBasis}, x::Array{TP, 1}) where {TP<:Real}
    grad = reshape(grad, size(fun.θ))
    fill!(grad, 0.)
    grad .= reshape(repeat(x, size(fun.θ)[end]), size(fun.θ))
end

function gradient!(grad::Array{TP}, fun::LinearFunction{TP, IdentityBasis}, x::Array{TP, 1}, y::Int) where {TP<:Real}
    grad = reshape(grad, size(fun.θ))
    fill!(grad, 0.)
    grad[:, y] .= x
end

function gradient!(grad::Array{TP}, fun::LinearFunction{TP}, x::Array{TP, 1}) where {TP<:Real}
    grad = reshape(grad, size(fun.θ))
    fill!(grad, 0.)
    call!(fun.feats, fun.ϕ, x)
    grad .= reshape(repeat(fun.feats, size(fun.θ)[end]), size(fun.θ))
end

function gradient!(grad::Array{TP}, fun::LinearFunction{TP}, x::Array{TP, 1}, y::Int) where {TP<:Real}
    grad = reshape(grad, size(fun.θ))
    fill!(grad, 0.)
    call!(fun.feats, fun.ϕ, x)
    grad[:, y] .= fun.feats
end

function gradient(fun::LinearFunction{TP}, x) where {TP<:Real}
    grad::Array{TP, 2} = zeros(TP, size(fun.θ))
    gradient!(grad, fun, x)
    return grad
end

function gradient(fun::LinearFunction{TP}, x, y::Int) where {TP<:Real}
    grad::Array{TP, 2} = zeros(TP, size(fun.θ))
    gradient!(grad, fun, x, y)
    return grad
end

function call_gradient!(out::Array{TP, 1}, grad::Array{TP}, fun::LinearFunction{TP, IdentityBasis}, x) where {TP<:Real}
    call!(out, fun, x)
    gradient!(grad, fun, x)
end

function call_gradient!(out::Array{TP, 1}, grad::Array{TP}, fun::LinearFunction{TP, IdentityBasis}, x::Array{TP, 1}) where {TP<:Real}
    call!(out, fun, x)
    gradient!(grad, fun, x)
end

function call_gradient!(grad::Array{TP}, fun::LinearFunction{TP, IdentityBasis}, x, y::Int)::TP where {TP<:Real}
    out::TP = call(fun, x, y)
    gradient!(grad, fun, x, y)
    return out
end

function call_gradient!(grad::Array{TP}, fun::LinearFunction{TP, IdentityBasis}, x::Array{TP, 1}, y::Int)::TP where {TP<:Real}
    out::TP = call(fun, x, y)
    gradient!(grad, fun, x, y)
    return out
end

function call_gradient!(out::Array{TP, 1}, grad::Array{TP}, fun::LinearFunction{TP}, x::Array{TP,1}) where {TP<:Real}
    call!(out, fun, x)
    g = reshape(grad, size(fun.θ))
    fill!(g, 0.)
    g .= reshape(repeat(fun.feats, size(fun.θ)[end]), size(fun.θ))
end

function call_gradient!(grad::Array{TP}, fun::LinearFunction{TP}, x::Array{TP, 1}, y::Int) where {TP<:Real}
    out = call(fun, x, y)
    grad = reshape(grad, size(fun.θ))
    fill!(grad, 0.)
    grad[:, y] .= fun.feats
    return out
end

function get_params(fun::LinearFunction{TP}) where {TP<:Real}
    return fun.θ
end

function copy_params!(params::Array{TP}, fun::LinearFunction{TP}) where {TP<:Real}
    p = reshape(params, size(fun.θ))
    copyto!(p, fun.θ)
end

function copy_params(fun::LinearFunction{TP}) where {TP<:Real}
    params = deepcopy(fun.θ)
    return params
end

function set_params!(fun::LinearFunction{TP}, θ) where {TP<:Real}
    fun.θ .= reshape(θ,size(fun.θ))
end

function add_to_params!(fun::LinearFunction{TP}, θ) where {TP<:Real}
    fun.θ .+= reshape(θ, size(fun.θ))
end

function get_num_params(fun::LinearFunction)::Int64
    return length(fun.θ)
end

function clone(fun::LinearFunction{TP, TB})::LinearFunction{TP, TB} where {TP<:Real,TB<:AbstractFuncApprox}
    fun2 = LinearFunction(deepcopy(fun.θ), clone(fun.ϕ), zeros(TP, size(fun.feats)))
    fun2.feats .= deepcopy(fun.feats)
    return fun2
end

function get_num_outputs(fun::LinearFunction{TP})::Int where {TP<:Real}
    return size(fun.θ)[end]
end
