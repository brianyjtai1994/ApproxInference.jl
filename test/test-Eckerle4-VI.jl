@testset "Variational Inference: NIST Dataset (Eckerle4)" begin
    print_head("Variational Inference: NIST Dataset (Eckerle4)")

    obj = optimizer(nd=3, ny=35, method="vi")
    fun = Eckerle4()
    sol = Vector{Float64}(undef, 3)

    parμ = [1.0, 10.0, 460.0]
    parΛ = zeros(Float64, 3, 3)
    βvec = zeros(Float64, 35)
    βmat = zeros(Float64, 35, 35)

    print_body("\033[1m\033[34mLikelihood Precision Matrix: Identity\033[0m")

    @simd for i in eachindex(βvec)
        @inbounds βvec[i] = 1.0
    end

    @simd for i in eachindex(βvec)
        @inbounds βmat[i,i] = 1.0
    end

    print_body("\033[35mPrior Precision Matrix: 1e-7⋅I\033[0m")

    @simd for i in eachindex(parμ)
        @inbounds parΛ[i,i] = 1e-7
    end

    ans = [1.55438763, 4.08885903, 451.541258]

    print_body(string("Precision Const.: ", optimize!(sol, fun, parμ, parΛ, 1.0, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    @inbounds ans[1], ans[2], ans[3] = 1.80359477, 5.81243698, 451.568834

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, parΛ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, parΛ, βmat, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body("\033[35mPrior Precision Matrix: 1e-5⋅I\033[0m")

    @simd for i in eachindex(parμ)
        @inbounds parΛ[i,i] = 1e-5
    end

    @inbounds ans[1], ans[2], ans[3] = 1.55487408, 4.09152161, 451.545157

    print_body(string("Precision Const.: ", optimize!(sol, fun, parμ, parΛ, 1.0, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    @inbounds ans[1], ans[2], ans[3] = 1.80680184, 5.84472833, 451.594154

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, parΛ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, parΛ, βmat, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body("\033[35mPrior Precision Matrix: 1e-3⋅I\033[0m")

    @simd for i in eachindex(parμ)
        @inbounds parΛ[i,i] = 1e-3
    end

    @inbounds ans[1], ans[2], ans[3] = 2.09790384, 11.1638318, 454.207767

    print_body(string("Precision Const.: ", optimize!(sol, fun, parμ, parΛ, 1.0, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    @inbounds ans[1], ans[2], ans[3] = 2.14766602, 10.7341140, 454.133335

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, parΛ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, parΛ, βmat, obj)))
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

    print_body("\033[35mPrior Precision Matrix: 1e-7⋅I\033[0m")

    @simd for i in eachindex(parμ)
        @inbounds parΛ[i,i] = 1e-7
    end

    @inbounds ans[1], ans[2], ans[3] = 1.78838264, 11.3022178, 456.415666

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, parΛ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, parΛ, βmat, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body("\033[35mPrior Precision Matrix: 1e-5⋅I\033[0m")

    @simd for i in eachindex(parμ)
        @inbounds parΛ[i,i] = 1e-5
    end

    @inbounds ans[1], ans[2], ans[3] = 1.78807741, 11.3010635, 456.420272

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, parΛ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, parΛ, βmat, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body("\033[35mPrior Precision Matrix: 1e-3⋅I\033[0m")

    @simd for i in eachindex(parμ)
        @inbounds parΛ[i,i] = 1e-3
    end

    @inbounds ans[1], ans[2], ans[3] = 2.14221961, 9.80389370, 447.594127

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, parΛ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, parΛ, βmat, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end
end
