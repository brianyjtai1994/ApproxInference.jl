# Levenberg-Marquardt step by 1st order linearization
function lmstep!(d::VecIO, A::MatI, β::Real, J::MatI, rs::VecI, Λ0::MatI, Δμ::VecI)
    _, cholesky_state = LAPACK.potrf!('L', A)
    BLAS.symv!('U', 1.0, Λ0, Δμ,  0.0, d) # d ← Λ0  * Δμ
    BLAS.gemv!('T',   β,  J, rs, -1.0, d) # d ← βJ' * rs - d
    BLAS.trsv!('L', 'N', 'N', A, d)
    BLAS.trsv!('L', 'T', 'N', A, d)
    return cholesky_state
end

function lmstep!(d::VecIO, A::MatI, ΛJ::MatI, rs::VecI, Λ0::MatI, Δμ::VecI)
    _, cholesky_state = LAPACK.potrf!('L', A)
    BLAS.symv!('U', 1.0, Λ0, Δμ,  0.0, d) # d ← Λ0  * Δμ
    BLAS.gemv!('T', 1.0, ΛJ, rs, -1.0, d) # d ← ΛJ' * rs - d
    BLAS.trsv!('L', 'N', 'N', A, d)
    BLAS.trsv!('L', 'T', 'N', A, d)
    return cholesky_state
end

# Geodesic acceleration step by 2nd order linearization
function lmstep!(d::VecIO, A::MatI, β::Real, J::MatI, rs::VecI)
    BLAS.gemv!('T', β, J, rs, 0.0, d)
    BLAS.trsv!('L', 'N', 'N', A, d)
    BLAS.trsv!('L', 'T', 'N', A, d)
    return nothing
end

function lmstep!(d::VecIO, A::MatI, ΛJ::MatI, rs::VecI)
    BLAS.gemv!('T', 1.0, ΛJ, rs, 0.0, d)
    BLAS.trsv!('L', 'N', 'N', A, d)
    BLAS.trsv!('L', 'T', 'N', A, d)
    return nothing
end
