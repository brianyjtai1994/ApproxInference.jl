# Get approximated hessian (posterior precision matrix)
function vi_get_apprhess!(H::MatIO, β::Real, J::MatI, Λ0::MatI)
    unsafe_copy!(H, Λ0) # H ← Λ0
    BLAS.syrk!('U', 'T', β, J, 1.0, H)
    @inbounds for j in axes(H, 1), i in 1:j-1
        H[j,i] = H[i,j]
    end
    return nothing
end

function vi_get_apprhess!(H::MatIO, λy::VecI, J::MatI, Λ0::MatI, ΛJ::MatB)
    for j in axes(J, 2)
        @simd for i in eachindex(λy)
            @inbounds ΛJ[i,j] = λy[i] * J[i,j]
        end
    end
    unsafe_copy!(H, Λ0)                      # H ← Λ0
    BLAS.gemm!('T', 'N', 1.0, ΛJ, J, 1.0, H) # H ← JΛ * J + Λ0
    return nothing
end

function vi_get_apprhess!(H::MatIO, Λy::MatI, J::MatI, Λ0::MatI, ΛJ::MatB)
    unsafe_copy!(H, Λ0)                       # H  ← Λ0
    BLAS.symm!('L', 'U', 1.0, Λy, J, 0.0, ΛJ) # ΛJ ← Λy * J
    BLAS.gemm!('T', 'N', 1.0, ΛJ, J, 1.0,  H) # H  ← JΛ * J + Λ0
    return nothing
end

# Get 1st-order variational inference gradient
function vi_get_gradient!(d::VecIO, β::Real, J::MatI, r::VecI, Λ0::MatI, Δμ::VecI)
    BLAS.symv!('U', 1.0, Λ0, Δμ,  0.0, d) # d ← Λ0  * Δμ
    BLAS.gemv!('T',   β,  J,  r, -1.0, d) # d ← βJ' * rs - Λ0  * Δμ
    return nothing
end

function vi_get_gradient!(d::VecIO, ΛJ::MatI, r::VecI, Λ0::MatI, Δμ::VecI)
    BLAS.symv!('U', 1.0, Λ0, Δμ,  0.0, d) # d ← Λ0  * Δμ
    BLAS.gemv!('T', 1.0, ΛJ,  r, -1.0, d) # d ← ΛJ' * rs - Λ0  * Δμ
    return nothing
end

# Perform BLAS.trsm!
function vi_get_blastrsm!(m::MatO, A::MatI)
    BLAS.trsm!('L', 'L', 'N', 'N', 1.0, A, m)
    BLAS.trsm!('L', 'L', 'T', 'N', 1.0, A, m)
    return nothing
end

# Get evidence (free energy) of variational inference
function vi_get_evidence!(Λy::Union{Real, VecI, MatI}, rs::VecI, ny::Int, Λ0::Union{Real, VecI, MatI}, Δμ::VecI, nd::Int, Fs::MatB, Λb::MatB)
    lm_get_cholesky!(Fs)
    vi_get_blastrsm!(Λb, Fs)
    return lm_get_squaresum(Λy, rs, ny) + lm_get_squaresum(Λ0, Δμ, nd) + 0.5 * (logdet(Fs, nd) + tr(Λb, nd))
end

struct VariationalInferenceOptimizer
    @def prop Vector{Float64} rs rt μs μt dμ δμ Δμ
    @def prop Matrix{Float64} Js Jt ΛJ Λs Λt Λb Ft Fb
    @def prop Int nd ny

    function VariationalInferenceOptimizer(nd::Int, ny::Int)
        @def vars Vector{Float64}(undef, ny) rs rt
        @def vars Vector{Float64}(undef, nd) μs μt dμ δμ Δμ
        @def vars Matrix{Float64}(undef, ny, nd) Js Jt ΛJ
        @def vars Matrix{Float64}(undef, nd, nd) Λs Λt Λb Ft Fb
        return new(rs, rt, μs, μt, dμ, δμ, Δμ, Js, Jt, ΛJ, Λs, Λt, Λb, Ft, Fb, nd, ny)
    end
end

function vi_get_optimize!(xs::VecIO, fn::AbstractObjective, μ0::VecI, Λ0::MatI, βy::Real, τ::Real, h::Real, itmax::Int, vi::VariationalInferenceOptimizer)
    #### Allocations ####
    @get vi nd ny rs rt μs μt dμ δμ Δμ
    @get vi Js Jt Λs Λt Λb Ft Fb
    #### Initialization
    ih1 = inv(h)
    ih2 = ih1 * ih1
    toN = eachindex(1:nd)
    unsafe_copy!(μs, μ0)
    unsafe_copy!(Λs, Λ0)
    #### Compute precision matrix
    lm_get_residual!(rs, μs, fn)
    lm_get_jacobian!(Js, μs, fn)
    vi_get_apprhess!(Λt, βy, Js, Λ0)
    # Controlling parameters of trust region
    a = τ * diagmax(Λt, nd); b = 2.0
    #### Compute free energy
    fill!(Δμ, 0.0)
    unsafe_copy!(Ft, Λs)
    unsafe_copy!(Fb, Λt)
    Fnow = vi_get_evidence!(βy, rs, ny, Λ0, Δμ, nd, Ft, Fb)
    #### Iteration
    it = 0
    while it < itmax
        it += 1
        # Damped Gauss-Newton approximation
        unsafe_copy!(Ft, Λt)
        λ = a * diagmax(Ft, nd)
        @simd for i in toN
            @inbounds Ft[i,i] += λ
        end
        cholesky_state = lm_get_cholesky!(Ft)
        # Levenberg-Marquardt step by 1st order linearization
        vi_get_gradient!(dμ, βy, Js, rs, Λ0, Δμ)
        lm_get_blastrsv!(dμ, Ft)
        # Finite-difference of 2nd order directional derivative
        @simd for i in toN
            @inbounds μt[i] = μs[i] + h * dμ[i]
        end
        lm_get_residual!(rt, μt, fn)
        lm_get_gradient!(rt, rs, Js, dμ, ih1, ih2)
        # Geodesic acceleration step by 2nd order linearization
        lm_get_gradient!(δμ, βy, Js, rt)
        lm_get_blastrsv!(δμ, Ft)
        gratio = 2.0 * nrm2(δμ) / nrm2(dμ)
        gratio > 0.75 && (a *= b; b *= 2.0; continue)
        # (Levenberg-Marquardt velocity) + (Geodesic acceleration)
        BLAS.axpy!(1.0, δμ, dμ) # dμ ← dμ + δμ
        # Compute predicted gain of the least square cost by linear approximation
        LinApprox = lm_get_squaresum(Λt, dμ, nd) + λ * dot(dμ, dμ, nd)
        # Compute gain ratio
        @simd for i in toN
            @inbounds μt[i] = μs[i] + dμ[i]
        end
        lm_get_residual!(rt, μt, fn)
        lm_get_jacobian!(Jt, μt, fn)
        vi_get_apprhess!(Λb, βy, Jt, Λ0)
        @simd for i in toN
            @inbounds δμ[i] = μt[i] - μ0[i]
        end
        unsafe_copy!(Ft, Λt)
        unsafe_copy!(Fb, Λb)
        Fnew = vi_get_evidence!(βy, rt, ny, Λ0, δμ, nd, Ft, Fb)

        Δ = Fnow - Fnew; ρ = Δ / LinApprox
        if ρ > 0.0
            Fnow = Fnew
            unsafe_copy!(μs, μt)
            unsafe_copy!(Λs, Λt)
            (maximum(δμ) < 1e-15 || Δ < 1e-10) && break # Check localized convergence
            unsafe_copy!(rs, rt)
            unsafe_copy!(Λt, Λb)
            unsafe_copy!(Js, Jt)
            unsafe_copy!(Δμ, δμ)
            a = ρ < 0.9367902323681495 ? a * (1.0 - cubic(2.0 * ρ - 1.0)) : a / 3.0
            b = 2.0
        else
            a > 1e10 && break # Check localized divergence
            a *= b; b *= 2.0
        end
    end
    unsafe_copy!(xs, μs); return xs
end

function vi_get_optimize!(xs::VecIO, fn::AbstractObjective, μ0::VecI, Λ0::MatI, Λy::Union{VecI, MatI}, τ::Real, h::Real, itmax::Int, vi::VariationalInferenceOptimizer)
    #### Allocations ####
    @get vi nd ny rs rt μs μt dμ δμ Δμ
    @get vi Js Jt ΛJ Λs Λt Λb Ft Fb
    #### Initialization
    ih1 = inv(h)
    ih2 = ih1 * ih1
    toN = eachindex(1:nd)
    unsafe_copy!(μs, μ0)
    unsafe_copy!(Λs, Λ0)
    #### Compute precision matrix
    lm_get_residual!(rs, μs, fn)
    lm_get_jacobian!(Js, μs, fn)
    vi_get_apprhess!(Λt, Λy, Js, Λ0, ΛJ) # ΛJ = Λy * Js + Λ0
    # Controlling parameters of trust region
    a = τ * diagmax(Λt, nd); b = 2.0
    #### Compute free energy
    fill!(Δμ, 0.0)
    unsafe_copy!(Ft, Λs)
    unsafe_copy!(Fb, Λt)
    Fnow = vi_get_evidence!(Λy, rs, ny, Λ0, Δμ, nd, Ft, Fb)
    #### Iteration
    it = 0
    while it < itmax
        it += 1
        # Damped Gauss-Newton approximation
        unsafe_copy!(Ft, Λt)
        λ = a * diagmax(Ft, nd)
        @simd for i in toN
            @inbounds Ft[i,i] += λ
        end
        cholesky_state = lm_get_cholesky!(Ft)
        # Levenberg-Marquardt step by 1st order linearization
        vi_get_gradient!(dμ, ΛJ, rs, Λ0, Δμ)
        lm_get_blastrsv!(dμ, Ft)
        # Finite-difference of 2nd order directional derivative
        @simd for i in toN
            @inbounds μt[i] = μs[i] + h * dμ[i]
        end
        lm_get_residual!(rt, μt, fn)
        lm_get_gradient!(rt, rs, Js, dμ, ih1, ih2)
        # Geodesic acceleration step by 2nd order linearization
        lm_get_gradient!(δμ, ΛJ, rt)
        lm_get_blastrsv!(δμ, Ft)
        gratio = 2.0 * nrm2(δμ) / nrm2(dμ)
        gratio > 0.75 && (a *= b; b *= 2.0; continue)
        # (Levenberg-Marquardt velocity) + (Geodesic acceleration)
        BLAS.axpy!(1.0, δμ, dμ) # dμ ← dμ + δμ
        # Compute predicted gain of the least square cost by linear approximation
        LinApprox = lm_get_squaresum(Λt, dμ, nd) + λ * dot(dμ, dμ, nd)
        # Compute gain ratio
        @simd for i in toN
            @inbounds μt[i] = μs[i] + dμ[i]
        end
        lm_get_residual!(rt, μt, fn)
        lm_get_jacobian!(Jt, μt, fn)
        vi_get_apprhess!(Λb, Λy, Jt, Λ0, ΛJ)
        @simd for i in toN
            @inbounds δμ[i] = μt[i] - μ0[i]
        end
        unsafe_copy!(Ft, Λt)
        unsafe_copy!(Fb, Λb)
        Fnew = vi_get_evidence!(Λy, rt, ny, Λ0, δμ, nd, Ft, Fb)

        Δ = Fnow - Fnew; ρ = Δ / LinApprox
        if ρ > 0.0
            Fnow = Fnew
            unsafe_copy!(μs, μt)
            unsafe_copy!(Λs, Λt)
            (maximum(δμ) < 1e-15 || Δ < 1e-10) && break # Check localized convergence
            unsafe_copy!(rs, rt)
            unsafe_copy!(Λt, Λb)
            unsafe_copy!(Js, Jt)
            unsafe_copy!(Δμ, δμ)
            a = ρ < 0.9367902323681495 ? a * (1.0 - cubic(2.0 * ρ - 1.0)) : a / 3.0
            b = 2.0
        else
            a > 1e10 && break # Check localized divergence
            a *= b; b *= 2.0
        end
    end
    unsafe_copy!(xs, μs); return xs
end

optimize!(xs::VecIO, fn::AbstractObjective, μ0::VecI, Λ0::MatI, Λy::Union{Real, VecI, MatI}, vi::VariationalInferenceOptimizer; τ::Real=1e-3, h::Real=0.1, itmax::Int=100) = vi_get_optimize!(xs, fn, μ0, Λ0, Λy, τ, h, itmax, vi)
