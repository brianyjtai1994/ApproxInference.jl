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

function ApproxInference.get_residual!(r::VecIO, ??::VecI, f::BoxBOD)
    x, y = f.x, f.y
    @inbounds ??1 = ??[1]
    @inbounds ??2 = ??[2]
    @simd for i in eachindex(x)
        @inbounds r[i] = y[i] - ??1 * (1.0 - exp(-??2 * x[i]))
    end
    return nothing
end

function ApproxInference.get_jacobian!(J::MatIO, ??::VecI, f::BoxBOD)
    x = f.x
    @inbounds ??1 = ??[1]
    @inbounds ??2 = ??[2]
    @inbounds for i in eachindex(x)
        tmp = exp(-??2 * x[i])
        J[i,1] = 1.0 - tmp
        J[i,2] = ??1 * x[i] * tmp
    end
    return nothing
end

function main(fname::String, ??0::VecI, ??0::MatI, ??y::Union{VecI, MatI}, ??::Real, h::Real, itmax::Int)
    fn = BoxBOD()
    vi = VariationalInferenceOptimizer(2, 6)
    x1 = Float64[]; @inbounds push!(x1, ??0[1])
    x2 = Float64[]; @inbounds push!(x2, ??0[2])
    #### Allocations ####
    @get vi nd ny rs rt ??s ??t d?? ???? ????
    @get vi Js Jt ??J ??s ??t ??b Ft Fb
    #### Initialization
    ih1 = inv(h)
    ih2 = ih1 * ih1
    toN = eachindex(1:nd)
    unsafe_copy!(??s, ??0)
    unsafe_copy!(??s, ??0)
    #### Compute precision matrix
    lm_get_residual!(rs, ??s, fn)
    lm_get_jacobian!(Js, ??s, fn)
    vi_get_apprhess!(??t, ??y, Js, ??0, ??J) # ??J = ??y * Js + ??0
    # Controlling parameters of trust region
    a = ?? * diagmax(??t, nd); b = 2.0
    #### Compute free energy
    fill!(????, 0.0)
    unsafe_copy!(Ft, ??s)
    unsafe_copy!(Fb, ??t)
    Fnow = vi_get_evidence!(??y, rs, ny, ??0, ????, nd, Ft, Fb)
    #### Iteration
    it = 0
    while it < itmax
        it += 1
        # Damped Gauss-Newton approximation
        unsafe_copy!(Ft, ??t)
        ?? = a * diagmax(Ft, nd)
        @simd for i in toN
            @inbounds Ft[i,i] += ??
        end
        cholesky_state = lm_get_cholesky!(Ft)
        # Levenberg-Marquardt step by 1st order linearization
        vi_get_gradient!(d??, ??J, rs, ??0, ????)
        lm_get_blastrsv!(d??, Ft)
        # Finite-difference of 2nd order directional derivative
        @simd for i in toN
            @inbounds ??t[i] = ??s[i] + h * d??[i]
        end
        lm_get_residual!(rt, ??t, fn)
        lm_get_gradient!(rt, rs, Js, d??, ih1, ih2)
        # Geodesic acceleration step by 2nd order linearization
        lm_get_gradient!(????, ??J, rt)
        lm_get_blastrsv!(????, Ft)
        gratio = 2.0 * nrm2(????) / nrm2(d??)
        gratio > 0.75 && (a *= b; b *= 2.0; continue)
        # (Levenberg-Marquardt velocity) + (Geodesic acceleration)
        BLAS.axpy!(1.0, ????, d??) # d?? ??? d?? + ????
        # Compute predicted gain of the least square cost by linear approximation
        LinApprox = lm_get_squaresum(??t, d??, nd) + ?? * dot(d??, d??, nd)
        # Compute gain ratio
        @simd for i in toN
            @inbounds ??t[i] = ??s[i] + d??[i]
        end
        lm_get_residual!(rt, ??t, fn)
        lm_get_jacobian!(Jt, ??t, fn)
        vi_get_apprhess!(??b, ??y, Jt, ??0, ??J)
        @simd for i in toN
            @inbounds ????[i] = ??t[i] - ??0[i]
        end
        unsafe_copy!(Ft, ??t)
        unsafe_copy!(Fb, ??b)
        Fnew = vi_get_evidence!(??y, rt, ny, ??0, ????, nd, Ft, Fb)

        ?? = Fnow - Fnew; ?? = ?? / LinApprox
        if ?? > 0.0
            Fnow = Fnew
            unsafe_copy!(??s, ??t)
            ###
            @inbounds push!(x1, ??s[1])
            @inbounds push!(x2, ??s[2])
            ###
            unsafe_copy!(??s, ??t)
            (maximum(????) < 1e-15 || ?? < 1e-10) && break # Check localized convergence
            unsafe_copy!(rs, rt)
            unsafe_copy!(??t, ??b)
            unsafe_copy!(Js, Jt)
            unsafe_copy!(????, ????)
            a = ?? < 0.9367902323681495 ? a * (1.0 - cubic(2.0 * ?? - 1.0)) : a / 3.0
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
        @inbounds ??s[1] = contourX[j]
        @inbounds for i in eachindex(contourY)
            @inbounds ??s[2] = contourY[i]
            lm_get_residual!(rs, ??s, fn)
            contourZ[i,j] = log10(lm_get_squaresum(??y, rs, ny))
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
    par?? = [10.0, 5.5]
    par?? = zeros(Float64, 2, 2)
    ??mat = zeros(Float64, 6, 6)

    @simd for i in eachindex(1:6)
        @inbounds ??mat[i,i] = 1.0
    end

    for ?? in (1e-7, 1e-5, 1e-3, 1e-1, 5e-1, 1e0)
        @simd for i in eachindex(par??)
            @inbounds par??[i,i] = ??
        end
        fname = @sprintf "./BoxBOD by Variational-Inference Method ??=%.1e.pdf" ??
        main(fname, par??, par??, ??mat, 1e-3, 0.1, 100)
    end

    return nothing
end

main()
