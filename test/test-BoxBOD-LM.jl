@testset "Levenberg-Marquardt: NIST Dataset (BoxBOD)" begin
    print_head("Levenberg-Marquardt: NIST Dataset (BoxBOD)")

    obj = optimizer(nd=2, ny=6, method="lm")
    fun = BoxBOD()
    sol = Vector{Float64}(undef, 2)

    parμ = [10.0, 5.5]
    βvec = zeros(Float64, 6)
    βmat = zeros(Float64, 6, 6)

    print_body("\033[1m\033[34mLikelihood Precision Matrix: Identity\033[0m")

    @simd for i in eachindex(βvec)
        @inbounds βvec[i] = 1.0
    end

    @simd for i in eachindex(βvec)
        @inbounds βmat[i,i] = 1.0
    end

    ans = [213.809409, 0.54723748]

    print_body(string("Precision Const.: ", optimize!(sol, fun, parμ, 1.0, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, βmat, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body("\033[1m\033[34mLikelihood Precision Matrix: Custom\033[0m")

    @simd for i in eachindex(βvec)
        @inbounds βvec[i] = 1.0 + 0.1 * (i-1)
    end

    @simd for i in eachindex(βvec)
        @inbounds βmat[i,i] = 1.0 + 0.1 * (i-1)
    end

    @inbounds ans[1], ans[2] = 216.132114, 0.52157598

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, βmat, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end
end
