module GradientCrystalPlasticity
using Tensors, Ferrite, StaticArrays
using FerriteAssembly

include("stardot.jl")
include("SlipSystems.jl")

struct GradCrystalPlasticity{T,N,D,M4,M2}
    E::SymmetricTensor{4,D,T,M4}
    ES::SMatrix{N,N,T}                       # [s⊗m]:E:[s⊗m]
    DS::SVector{N,SymmetricTensor{2,D,T,M2}} # [s⊗m]:E
    HGND::NTuple{2,T}                        # HGND[1] = self - cross, and HGND[2] = cross
    τy::T # Initial critically resolved shear stress
    t_star::T
    n_exp::T
end
function GradCrystalPlasticity(crystal::Crystallography;
        elastic_stiffness::SymmetricTensor{4}, 
        yield_limit,
        self_hardening, cross_hardening,
        t_star, n_exp)
    dyads = get_slip_dyads(crystal)
    N = get_num_slipsystems(crystal)
    DS = map(dyad -> dyad ⊡ elastic_stiffness, dyads)
    ES = SMatrix{N,N}((DS_i ⊡ dyad_j for DS_i in DS, dyad_j in dyads)...)
    HGND = (self_hardening - cross_hardening, cross_hardening)
    return GradCrystalPlasticity(elastic_stiffness, ES, DS, HGND, yield_limit, t_star, n_exp)
end

slipnames(::GradCrystalPlasticity{<:Any, 2}) = (:γ1, :γ2)
slipnames(::GradCrystalPlasticity{<:Any,12}) = (:γ1, :γ2, :γ3, :γ4, :γ5, :γ6, :γ7, :γ8, :γ9, :γ10, :γ11, :γ12)
crssnames(::GradCrystalPlasticity{<:Any, 2}) = (:τ1, :τ2)
crssnames(::GradCrystalPlasticity{<:Any,12}) = (:τ1, :τ2, :τ3, :τ4, :τ5, :τ6, :τ7, :τ8, :τ9, :τ10, :τ11, :τ12)


function FerriteAssembly.element_residual!(re, state, ae, m::GradCrystalPlasticity{T,N,dim}, cv, buffer) where {T,N,dim}
    slipkeys = slipnames(m)
    crsskeys = crssnames(m)
    ae_old = FerriteAssembly.get_aeold(buffer)
    Δt = FerriteAssembly.get_time_increment(buffer)
    for q_point in 1:getnquadpoints(cv[:u])
        dΩ = getdetJdV(cv[:u], q_point)
        ϵ = function_symmetric_gradient(cv[:u], q_point, ae, dof_range(buffer, :u))
        γ = SVector(map(k -> function_value(cv[:γ], q_point, ae, dof_range(buffer, k)), slipkeys))
        γold = SVector(map(k -> function_value(cv[:γ], q_point, ae_old, dof_range(buffer, k)), slipkeys))
        ∇γ = SVector(map(k -> function_gradient(cv[:γ], q_point, ae, dof_range(buffer, k)), slipkeys))
        ∇γ_sum = sum(∇γ)
        σ = m.E ⊡ ϵ - m.DS ⋆ γ
        τᵈⁱ = SVector(map(k -> function_value(cv[:τ], q_point, ae, dof_range(buffer, k)), crsskeys))

        # Displacement field, u
        for (iᵤ, Iᵤ) in pairs(dof_range(buffer, :u))
            ∇δN = shape_symmetric_gradient(cv[:u], q_point, iᵤ)
            re[Iᵤ] += (∇δN⊡σ)*dΩ
        end
        # Slip fields, γ
        for (slipnr, slipkey) in pairs(slipkeys)
            γ_L2_part = (- m.DS[slipnr]⊡ϵ + m.ES[slipnr, :] ⋆ γ + τᵈⁱ[slipnr]) * dΩ
            γ_H1_part = (m.HGND[1] * ∇γ[slipnr] + m.HGND[2] * ∇γ_sum) * dΩ

            for (iγ, Iγ) in pairs(dof_range(buffer, slipkey))
                δNγ = shape_value(cv[:γ], q_point, iγ)
                ∇δNγ = shape_gradient(cv[:γ], q_point, iγ)
                re[Iγ] += δNγ * γ_L2_part + ∇δNγ ⋅ γ_H1_part
            end
        end
        # Resolved stress fields, τ
        for (slipnr, crsskey) in pairs(crsskeys)
            γdot = (γ[slipnr] - γold[slipnr])/Δt
            ϕ = (sign(τᵈⁱ[slipnr])/m.t_star) * ((sqrt(τᵈⁱ[slipnr]^2) - m.τy)/m.τy)^m.n_exp
            τ_part = (γdot - ϕ) * dΩ

            for (iτ, Iτ) in pairs(dof_range(buffer, crsskey))
                δNτ = shape_value(cv[:τ], q_point, iτ)
                re[Iτ] += δNτ * τ_part
            end
        end

    end
end

end
