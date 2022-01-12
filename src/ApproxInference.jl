module ApproxInference

const VecI  = AbstractVector # Input  Vector
const VecO  = AbstractVector # Output Vector
const VecB  = AbstractVector # Buffer Vector
const VecIO = AbstractVector # In/Out Vector
const MatI  = AbstractMatrix # Input  Matrix
const MatO  = AbstractMatrix # Output Matrix
const MatB  = AbstractMatrix # Buffer Matrix
const MatIO = AbstractMatrix # In/Out Matrix

import LinearAlgebra: BLAS
import LinearAlgebra: LAPACK

if haskey(ENV, "BLAS_THREAD_NUM")
    BLAS.set_num_threads(parse(Int, ENV["BLAS_THREAD_NUM"]))
else
    BLAS.set_num_threads(4)
end

include("./linalg.jl")
include("./macros.jl")
include("./statistics.jl")

abstract type AbstractObjective <: Function end

macro objective(e::Expr)
    e.head ≡ :struct || error("Objective funtion should be defined as `struct ... end`.")
    @inbounds a = e.args[2]
    if typeof(a) <: Symbol
        @inbounds e.args[2] = Expr(:<:, a, :AbstractObjective)
    else
        @inbounds a.args[2] = :AbstractObjective
    end
    return e
end

function optimize!     end
function get_residual! end
function get_jacobian! end

include("./lmpack.jl")

function optimizer(nd::Int, ny::Int; method::String="lm")
    if method ≡ "lm" || method ≡ "LM" || method ≡ "LevenbergMarquardt" || method ≡ "Levenberg-Marquardt"
        return LevenbergMarquardtOptimizer(nd, ny)
    end
end

export @objective, optimize!, get_residual!, get_jacobian!, optimizer

end # module
