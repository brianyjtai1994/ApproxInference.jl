export get_residual!, get_jacobian!

function get_residual! end
function get_jacobian! end

# Get residuals: (predicted - observed) data
lm_get_residual!(r::VecIO, μ::VecI, f::Function) = get_residual!(r, μ, f)

# Get jacobian matrix of predicted model
lm_get_jacobian!(J::MatIO, μ::VecI, f::Function) = get_jacobian!(J, μ, f)

# Get approximated hessian of least-square residuals
function lm_get_apprhess!(H::MatIO, β::Real, J::MatI)
    BLAS.syrk!('U', 'T', β, J, 0.0, H)
    @inbounds for j in axes(H, 1), i in 1:j-1
        H[j,i] = H[i,j]
    end
    return nothing
end

function lm_get_apprhess!(H::MatIO, Λy::MatI, J::MatI, ΛJ::MatB)
    BLAS.symm!('L', 'U', 1.0, Λy, J, 0.0, ΛJ) # ΛJ ← Λy * J
    BLAS.gemm!('T', 'N', 1.0, ΛJ, J, 0.0,  H) # H  ← JΛ * J
    return nothing
end

# Get 1st-order least-square gradient
lm_get_gradient!(d::VecIO, β::Real, J::MatI, r::VecI) = BLAS.gemv!('T', β, J, r, 0.0, d)
lm_get_gradient!(d::VecIO, ΛJ::MatI, r::VecI)         = BLAS.gemv!('T', 1.0, ΛJ, r, 0.0, d)

# Get 2nd-order directional gradient
function lm_get_gradient!(n::VecIO, r::VecI, J::MatI, d::VecI, ih1::Real, ih2::Real)
    BLAS.axpy!(-1.0, r, n)
    BLAS.gemv!('N', ih1, J, d, ih2, n)
    return nothing
end

function lm_get_cholesky!(A::MatIO)
    _, cholesky_state = LAPACK.potrf!('L', A)
    return cholesky_state
end

# Perform BLAS.trsv!
function lm_get_blastrsv!(d::VecIO, A::MatIO)
    BLAS.trsv!('L', 'N', 'N', A, d)
    BLAS.trsv!('L', 'T', 'N', A, d)
    return nothing
end

struct LMOptimizer
    @def prop Vector{Float64} rs rt μs μt dμ δμ
    @def prop Matrix{Float64} Js Jt As Hs Hf
    @def prop Int nd ny

    function LMOptimizer(nd::Int, ny::Int)
        @def vars Vector{Float64}(undef, ny) rs rt
        @def vars Vector{Float64}(undef, nd) μs μt dμ δμ
        @def vars Matrix{Float64}(undef, ny, nd) Js Jt As
        @def vars Matrix{Float64}(undef, nd, nd) Hs Hf
        return new(rs, rt, μs, μt, dμ, δμ, Js, Jt, As, Hs, Hf, nd, ny)
    end
end

# *t := temp one of *
# *s := solution of *
# Hf := hessian for factorization
function lm_get_optimize!(o::LMOptimizer, fn::Function, μ0::VecI, βy::Real; τ::Real=1e-3, h::Real=0.1, itmax::Int=100)
    #### Allocations ####
    @get o nd ny rs rt μs μt dμ δμ
    @get o Js Jt Hs Hf
    #### Initialization
    ih1 = inv(h)
    ih2 = ih1 * ih1
    toN = eachindex(1:nd)
    unsafe_copy!(μs, μ0)
    #### Compute precision matrix
    lm_get_residual!(rs, μs, fn)
    lm_get_jacobian!(Js, μs, fn)
    lm_get_apprhess!(Hs, βy, Js)
    # Controlling Parameters of Trust Region
    a = τ * diagmax(Hs, nd); b = 2.0
    #### Compute free energy
    Cnow = 0.5 * βy * dot(rs, rs, ny)
    #### Iteration
    it = 0
    while it < itmax
        it += 1
        # Damped Gauss-Newton approximation
        unsafe_copy!(Hf, Hs)
        λ = a * diagmax(Hf, nd)
        @simd for i in toN
            @inbounds Hf[i,i] += λ
        end
        cholesky_state = lm_get_cholesky!(Hf)
        # Levenberg-Marquardt step by 1st order linearization
        lm_get_gradient!(dμ, βy, Js, rs)
        lm_get_blastrsv!(dμ, Hf)
        # Finite-difference of 2nd order directional derivative
        @simd for i in toN
            @inbounds μt[i] = μs[i] + h * dμ[i]
        end
        lm_get_residual!(rt, μt, fn)
        lm_get_gradient!(rt, rs, Js, dμ, ih1, ih2)
        # Geodesic acceleration step by 2nd order linearization
        lm_get_gradient!(δμ, βy, Js, rt)
        lm_get_blastrsv!(δμ, Hf)
        gratio = 2.0 * nrm2(δμ) / nrm2(dμ)
        gratio > 0.75 && (a *= b; b *= 2.0; continue)
        # (Levenberg-Marquardt velocity) + (Geodesic acceleration)
        BLAS.axpy!(1.0, δμ, dμ) # dμ ← dμ + δμ
        # Compute predicted gain of the least square cost by linear approximation
        LinApprox = 0.5 * dot(dμ, nd, Hs, dμ, nd) + λ * dot(dμ, dμ, nd)
        # Compute gain ratio
        @simd for i in toN
            @inbounds μt[i] = μs[i] + dμ[i]
        end
        lm_get_residual!(rt, μt, fn)
        Cnew = 0.5 * βy * dot(rt, rt, ny)

        Δ = Cnow - Cnew; ρ = Δ / LinApprox
        if ρ > 0.0
            Cnow = Cnew
            unsafe_copy!(μs, μt)
            (maximum(δμ) < 1e-15 || Δ < 1e-10) && break # Check localized convergence
            lm_get_jacobian!(Js, μs, fn)
            lm_get_apprhess!(Hs, βy, Js)
            unsafe_copy!(rs, rt)
            a = ρ < 0.9367902323681495 ? a * (1.0 - cubic(2.0 * ρ - 1.0)) : a / 3.0
            b = 2.0
        else
            a > 1e10 && break # Check localized divergence
            a *= b; b *= 2.0
        end
    end
    return nothing
end

function lm_get_optimize!(o::LMOptimizer, fn::Function, μ0::VecI, Λy::MatI; τ::Real=1e-3, h::Real=0.1, itmax::Int=100)
    #### Allocations ####
    @get o nd ny rs rt μs μt dμ δμ
    @get o Js Jt As Hs Hf
    #### Initialization
    ih1 = inv(h)
    ih2 = ih1 * ih1
    toN = eachindex(1:nd)
    unsafe_copy!(μs, μ0)
    #### Compute precision matrix
    lm_get_residual!(rs, μs, fn)
    lm_get_jacobian!(Js, μs, fn)
    lm_get_apprhess!(Hs, Λy, Js, As) # As = Λy * Js
    # Controlling Parameters of Trust Region
    a = τ * diagmax(Hs, nd); b = 2.0
    #### Compute free energy
    Cnow = 0.5 * dot(rs, ny, Λy, rs, ny)
    #### Iteration
    it = 0
    while it < itmax
        it += 1
        # Damped Gauss-Newton approximation
        unsafe_copy!(Hf, Hs)
        λ = a * diagmax(Hf, nd)
        @simd for i in toN
            @inbounds Hf[i,i] += λ
        end
        cholesky_state = lm_get_cholesky!(Hf)
        # Levenberg-Marquardt step by 1st order linearization
        lm_get_gradient!(dμ, As, rs)
        lm_get_blastrsv!(dμ, Hf)
        # Finite-difference of 2nd order directional derivative
        @simd for i in toN
            @inbounds μt[i] = μs[i] + h * dμ[i]
        end
        lm_get_residual!(rt, μt, fn)
        lm_get_gradient!(rt, rs, Js, dμ, ih1, ih2)
        # Geodesic acceleration step by 2nd order linearization
        lm_get_gradient!(δμ, As, rt)
        lm_get_blastrsv!(δμ, Hf)
        gratio = 2.0 * nrm2(δμ) / nrm2(dμ)
        gratio > 0.75 && (a *= b; b *= 2.0; continue)
        # (Levenberg-Marquardt velocity) + (Geodesic acceleration)
        BLAS.axpy!(1.0, δμ, dμ) # dμ ← dμ + δμ
        # Compute predicted gain of the least square cost by linear approximation
        LinApprox = 0.5 * dot(dμ, nd, Hs, dμ, nd) + λ * dot(dμ, dμ, nd)
        # Compute gain ratio
        @simd for i in toN
            @inbounds μt[i] = μs[i] + dμ[i]
        end
        lm_get_residual!(rt, μt, fn)
        Cnew = 0.5 * dot(rt, ny, Λy, rt, ny)

        Δ = Cnow - Cnew; ρ = Δ / LinApprox
        if ρ > 0.0
            Cnow = Cnew
            unsafe_copy!(μs, μt)
            (maximum(δμ) < 1e-15 || Δ < 1e-10) && break # Check localized convergence
            lm_get_jacobian!(Js, μs, fn)
            lm_get_apprhess!(Hs, Λy, Js, As)
            unsafe_copy!(rs, rt)
            a = ρ < 0.9367902323681495 ? a * (1.0 - cubic(2.0 * ρ - 1.0)) : a / 3.0
            b = 2.0
        else
            a > 1e10 && break # Check localized divergence
            a *= b; b *= 2.0
        end
    end
    return nothing
end
