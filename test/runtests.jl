using Test, ApproxInference

const VecI  = AbstractVector # Input  Vector
const VecIO = AbstractVector # In/Out Vector
const MatI  = AbstractMatrix # Input  Matrix
const MatIO = AbstractMatrix # In/Out Matrix

print_head(s::String) = println("\033[1m\033[32mTesting Task\033[0m \033[1m\033[33m$s\033[0m")
print_body(s::String) = println("             $s")

##############################
#   NIST Dataset: BoxBOD     #
##############################
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

include("./test-BoxBOD-LM.jl")
include("./test-BoxBOD-VI.jl")

##############################
#   NIST Dataset: Eckerle4   #
##############################
@objective struct Eckerle4
    x::Vector{Float64}; y::Vector{Float64}

    Eckerle4() = new(
        # x vector
        [400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0,
         435.0, 436.5, 438.0, 439.5, 441.0, 442.5, 444.0,
         445.5, 447.0, 448.5, 450.0, 451.5, 453.0, 454.5,
         456.0, 457.5, 459.0, 460.5, 462.0, 463.5, 465.0,
         470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 500.0],
        # y vector
        [1.575000e-04, 1.699000e-04, 2.350000e-04, 3.102000e-04, 4.917000e-04,
         8.710000e-04, 1.741800e-03, 4.640000e-03, 6.589500e-03, 9.730200e-03,
         1.490020e-02, 2.373100e-02, 4.016830e-02, 7.125590e-02, 1.264458e-01,
         2.073413e-01, 2.902366e-01, 3.445623e-01, 3.698049e-01, 3.668534e-01,
         3.106727e-01, 2.078154e-01, 1.164354e-01, 6.167640e-02, 3.372000e-02,
         1.940230e-02, 1.178310e-02, 7.435700e-03, 2.273200e-03, 8.800000e-04,
         4.579000e-04, 2.345000e-04, 1.586000e-04, 1.143000e-04, 7.100000e-05]
    )
end

function ApproxInference.get_residual!(r::VecIO, θ::VecI, f::Eckerle4)
    x, y = f.x, f.y
    @inbounds θ1 = θ[1]
    @inbounds θ2 = θ[2]
    @inbounds θ3 = θ[3]
    @simd for i in eachindex(x)
        @inbounds r[i] = y[i] - θ1 * exp(-0.5 * abs2((x[i] - θ3) / θ2)) / θ2
    end
    return nothing
end

function ApproxInference.get_jacobian!(J::MatIO, θ::VecI, f::Eckerle4)
    x = f.x
    @inbounds θ1 = θ[1]
    @inbounds θ2 = θ[2]
    @inbounds θ3 = θ[3]

    θ2² = abs2(θ2)
    θ2³ = θ2 * θ2 * θ2

    @inbounds for i in eachindex(x)
        tmp1 = abs2(x[i] - θ3) / θ2²
        tmp2 = exp(-0.5 * tmp1)
        J[i,1] = tmp2 / θ2
        J[i,2] = θ1 * tmp2 * (tmp1 - 1.0) / θ2²
        J[i,3] = θ1 * (x[i] - θ3) * tmp2 / θ2³
    end
    return nothing
end

include("./test-Eckerle4-LM.jl")
include("./test-Eckerle4-VI.jl")
