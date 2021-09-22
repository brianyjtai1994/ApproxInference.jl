function mean!(μ::VecIO, r::VecI, X::MatI)
    N = sum(r)
    if iszero(N)
        @simd for m in eachindex(μ)
            @inbounds μ[m] = 0.0
        end
    else
        BLAS.gemv!('T', 1.0, X, r, 0.0, μ)
        @simd for m in eachindex(μ)
            @inbounds μ[m] /= N
        end
    end
    return nothing
end

function mean!(μ::MatIO, R::MatI, X::MatI, N::VecI; init::Bool=false)
    if init
        @inbounds for i in eachindex(N)
            N[i] = sum(view(R,i,:))
        end
    end
    BLAS.gemm!('N', 'N', 1.0, R, X, 0.0, μ)
    for k in axes(μ, 1)
        @inbounds Nk = N[k]
        if iszero(Nk)
            @simd for m in axes(μ, 2)
                @inbounds μ[k,m] = 0.0
            end
        else
            @simd for m in axes(μ, 2)
                @inbounds μ[k,m] /= Nk
            end
        end
    end
    return nothing
end

function deviation!(Δ::MatIO, X::MatI, μ::VecI)
    n, m = size(X)
    for j in eachindex(1:m)
        @inbounds μj = μ[j]
        @simd for i in eachindex(1:n)
            @inbounds Δ[i,j] = X[i,j] - μj
        end
    end
    return nothing
end

function covariance!(C::MatIO, X::MatI, μ::VecI, r::VecI, Δ::MatB)
    deviation!(Δ, X, μ)
    for n in eachindex(r)
        @inbounds rn = sqrt(r[n])
        @simd for m in eachindex(μ)
            @inbounds Δ[n,m] *= rn
        end
    end
    BLAS.syrk!('U', 'T', 1.0, Δ, 0.0, C)
    @inbounds for j in eachindex(μ), i in 1:j-1
        C[j,i] = C[i,j]
    end
    return nothing
end
