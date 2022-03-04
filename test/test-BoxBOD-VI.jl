@testset "Variational Inference: NIST Dataset (BoxBOD)" begin
    print_head("Variational Inference: NIST Dataset (BoxBOD)")

    obj = optimizer(nd=2, ny=6, method="vi")
    fun = BoxBOD()
    sol = Vector{Float64}(undef, 2)

    parμ = [10.0, 5.5]
    parΛ = zeros(Float64, 2, 2)
    βvec = zeros(Float64, 6)
    βmat = zeros(Float64, 6, 6)

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

    ans = [213.809397, 0.54723757]

    print_body(string("Precision Const.: ", optimize!(sol, fun, parμ, parΛ, 1.0, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

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

    @inbounds ans[1], ans[2] = 213.808190, 0.54724585

    print_body(string("Precision Const.: ", optimize!(sol, fun, parμ, parΛ, 1.0, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

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

    @inbounds ans[1], ans[2] = 213.687749, 0.54807289

    print_body(string("Precision Const.: ", optimize!(sol, fun, parμ, parΛ, 1.0, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

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

    @inbounds ans[1], ans[2] = 177.894176, 10.761672

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

    @inbounds ans[1], ans[2] = 216.131185, 0.52158226

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

    @inbounds ans[1], ans[2] = 216.039327, 0.52220407

    print_body(string("Precision Vector: ", optimize!(sol, fun, parμ, parΛ, βvec, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end

    print_body(string("Precision Matrix: ", optimize!(sol, fun, parμ, parΛ, βmat, obj)))
    @inbounds for i in eachindex(ans)
        @test sol[i] ≈ ans[i] rtol=1e-7
    end
end
