const VecI  = AbstractVector # Input  Vector
const VecO  = AbstractVector # Output Vector
const VecB  = AbstractVector # Buffer Vector
const VecIO = AbstractVector # In/Out Vector
const MatI  = AbstractMatrix # Input  Matrix
const MatO  = AbstractMatrix # Output Matrix
const MatB  = AbstractMatrix # Buffer Matrix
const MatIO = AbstractMatrix # In/Out Matrix

using Printf

import LinearAlgebra: BLAS
BLAS.set_num_threads(4)

import PyPlot as plt
import ApproxInference
import ApproxInference: @get, @objective, unsafe_copy!, diagmax, nrm2, dot, cubic,
                        lm_get_residual!, lm_get_jacobian!, vi_get_apprhess!,
                        lm_get_gradient!, lm_get_cholesky!, lm_get_blastrsv!,
                        lm_get_squaresum, vi_get_evidence!, vi_get_gradient!,
                        VariationalInferenceOptimizer

meshgrid(x::VecI{T}, y::VecI{T}) where T = Matrix{T}(undef, length(y), length(x))

function init_rc!()
    rcParams = plt.PyDict(plt.matplotlib."rcParams")
    pgfpreamble = string("\\usepackage{amsthm}",
                         "\\usepackage{mathtools}",
                         "\\usepackage[libertine]{newtxmath}",
                         "\\usepackage[tt=false]{libertine}",
                         "\\usepackage{siunitx}",
                         "\\usepackage{mhchem}")
    # Backend
    rcParams["backend"] = "qt5cairo"
    # Latex
    rcParams["pgf.rcfonts"] = false
    rcParams["pgf.preamble"] = pgfpreamble
    rcParams["text.usetex"] = true
    rcParams["text.latex.preamble"] = "\\usepackage{siunitx}\\usepackage{mhchem}"
    # Font
    rcParams["font.size"] = 14
    rcParams["font.family"] = "serif"
    # Axes
    rcParams["axes.linewidth"] = 1.5
    rcParams["xtick.major.width"] = 1.5
    rcParams["ytick.major.width"] = 1.5
    # Grid
    rcParams["grid.alpha"] = 0.5
    rcParams["grid.linewidth"] = 1.5
    # Lines
    rcParams["lines.linewidth"] = 2.0
    # Legend
    rcParams["legend.framealpha"] = 0.0
    return nothing
end

@objective struct BoxBOD
    x::Vector{Float64}; y::Vector{Float64}

    BoxBOD() = new([1., 2., 3., 5., 7., 10.], [109., 149., 149., 191., 213., 224.])
end

function ApproxInference.get_residual!(r::VecIO, θ::VecI, f::BoxBOD)
    x, y = f.x, f.y
    @inbounds θ1 = θ[1]
    @inbounds θ2 = θ[2]
    @simd for i in eachindex(x)
        @inbounds r[i] = y[i] - θ1 * (1.0 - exp(-θ2 * x[i]))
    end
    return nothing
end

function ApproxInference.get_jacobian!(J::MatIO, θ::VecI, f::BoxBOD)
    x = f.x
    @inbounds θ1 = θ[1]
    @inbounds θ2 = θ[2]
    @inbounds for i in eachindex(x)
        tmp = exp(-θ2 * x[i])
        J[i,1] = 1.0 - tmp
        J[i,2] = θ1 * x[i] * tmp
    end
    return nothing
end

function main(fname::String, μ0::VecI, Λ0::MatI, Λy::Union{VecI, MatI}, τ::Real, h::Real, itmax::Int)
    fn = BoxBOD()
    vi = VariationalInferenceOptimizer(2, 6)
    x1 = Float64[]; @inbounds push!(x1, μ0[1])
    x2 = Float64[]; @inbounds push!(x2, μ0[2])
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
            ###
            @inbounds push!(x1, μs[1])
            @inbounds push!(x2, μs[2])
            ###
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

    contourX = collect(range(-50.0, 250.0; step=1.0)) # length(X) = N
    contourY = collect(range(-0.10, 14.0; step=0.05)) # length(Y) = M
    contourZ = meshgrid(contourX, contourY)           # size(Z) = (M, N)

    for j in eachindex(contourX)
        @inbounds μs[1] = contourX[j]
        @inbounds for i in eachindex(contourY)
            @inbounds μs[2] = contourY[i]
            lm_get_residual!(rs, μs, fn)
            contourZ[i,j] = log10(lm_get_squaresum(Λy, rs, ny))
        end
    end

    init_rc!()

    _cmap = plt.get_cmap("GnBu")
    fig   = plt.figure(num=1, figsize=(8.8, 6.6), clear=true, linewidth=6.)
    ax1   = fig.add_subplot(111)
    _fill = ax1.contourf(contourX, contourY, contourZ, cmap=_cmap, levels=50)
    _line = ax1.plot(x1, x2, "--*", c="k", lw=1)

    ax1.set_xlabel(raw"$\theta_{1}$")
    ax1.set_ylabel(raw"$\theta_{2}$")

    fig.colorbar(_fill, pad=0.02)
    ax1.patch.set_alpha(0.)
    fig.patch.set_alpha(0.)
    fig.set_tight_layout(true)
    fig.savefig(fname, format="pdf", dpi=600, backend="pgf", bbox_inches="tight", pad_inches=0.05)
    return nothing
end

function main()
    parμ = [10.0, 5.5]
    parΛ = zeros(Float64, 2, 2)
    βmat = zeros(Float64, 6, 6)

    @simd for i in eachindex(1:6)
        @inbounds βmat[i,i] = 1.0
    end

    for λ in (1e-7, 1e-5, 1e-3, 1e-1, 5e-1, 1e0)
        @simd for i in eachindex(parμ)
            @inbounds parΛ[i,i] = λ
        end
        fname = @sprintf "./BoxBOD by Variational-Inference Method λ=%.1e.pdf" λ
        main(fname, parμ, parΛ, βmat, 1e-3, 0.1, 100)
    end

    return nothing
end

main()
