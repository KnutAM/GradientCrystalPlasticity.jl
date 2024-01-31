using GradientCrystalPlasticity
using Test
using LinearAlgebra, StaticArrays, Tensors
import GradientCrystalPlasticity as GCP
import GradientCrystalPlasticity: ⋆

@testset "SlipSystems" begin
    for CT in (GCP.BCC, GCP.FCC, GCP.BCC12)
        @testset "$CT" begin
            crystal = CT()
            m = GCP.get_slip_planes(crystal)
            s = GCP.get_slip_directions(crystal)
            for (mα, sα) in zip(m, s)
                # Test perpendicular
                @test abs(mα⋅sα) < 1e-10
                # Test unit length
                @test norm(mα) ≈ 1
                @test norm(sα) ≈ 1
                # Test no duplicates
                num_equal = 0
                for (mma, ssa) in zip(m, s)
                    if (mma ≈ mα || mma ≈ -mα) && (ssa ≈ sα || ssa ≈ -sα)
                        num_equal += 1
                    end
                end
                @test num_equal == 1
            end
        end
    end
end

@testset "stardot" begin
    s1 = rand(SVector{4})
    s2 = rand(SVector{4})
    v1 = SVector{4}((rand(Vec{3}) for _ in 1:4))
    v2 = SVector{4}((rand(Vec{3}) for _ in 1:4))
    t1 = SVector{4}((rand(SymmetricTensor{2,3}) for _ in 1:4))
    @test s1 ⋆ s2 ≈ sum(s1 .* s2)
    @test s1 ⋆ v1 ≈ sum(s1 .* v1)
    @test s1 ⋆ t1 ≈ sum(s1 .* t1)
    @test v1 ⋆ v2 ≈ sum(v1 .⋅ v2)
    # Operator precedence
    @test (s1 ⋆ s2 + 1.0) ≈ sum(s1 .* s2) + 1.0
end
