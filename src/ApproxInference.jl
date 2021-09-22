module ApproxInference

const VecI  = AbstractVector # Input  Vector
const VecO  = AbstractVector # Output Vector
const VecB  = AbstractVector # Buffer Vector
const VecIO = AbstractVector # In/Out Vector
const MatI  = AbstractMatrix # Input  Matrix
const MatO  = AbstractMatrix # Output Matrix
const MatB  = AbstractMatrix # Buffer Matrix
const MatIO = AbstractMatrix # In/Out Matrix

const BLAS_THREAD_NUM = 4

import LinearAlgebra: BLAS

BLAS.set_num_threads(BLAS_THREAD_NUM)

include("./statistics.jl")

end # module
