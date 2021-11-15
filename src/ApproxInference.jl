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

if haskey(ENV, "BLAS_THREAD_NUM")
    BLAS.set_num_threads(parse(Int, ENV["BLAS_THREAD_NUM"]))
else
    BLAS.set_num_threads(4)
end

include("./statistics.jl")

end # module
