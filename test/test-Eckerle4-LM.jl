@testset "Levenberg-Marquardt: NIST Dataset (Eckerle4)" begin
    print_head("Levenberg-Marquardt: NIST Dataset (Eckerle4)")

    obj = optimizer(nd=3, ny=35, method="lm")
    fun = Eckerle4()
    sol = Vector{Float64}(undef, 3)

    parμ = [1.0, 10.0, 460.0]
    βvec = zeros(Float64, 35)
    βmat = zeros(Float64, 35, 35)

    print_body("\033[1m\033[34mLikelihood Precision Matrix: Identity\033[0m")

    @simd for i in eachindex(βvec)
        @inbounds βvec[i] = 1.0
    end

    @simd for i in eachindex(βvec)
        @inbounds βmat[i,i] = 1.0
    end

    ans = [1.55438286, 4.08883291, 451.541218]

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

    @inbounds ans[1], ans[2], ans[3] = 1.55104316, 4.07147586, 451.543588

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, βmat, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end
end
