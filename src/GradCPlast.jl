
struct GradCPlast{T,N,D,M4,M2}
    E::SymmetricTensor{4,D,T,M4}
    ES::SMatrix{N,N,T}                       # [s⊗m]:E:[s⊗m]
    DS::SVector{N,SymmetricTensor{2,D,T,M2}} # [s⊗m]:E
    HGND::NTuple{2,T}                        # HGND[1] = self - cross, and HGND[2] = cross
    τy::T # Initial critically resolved shear stress
    t_star::T
    n_exp::T
end
function GradCPlast(crystal::Crystallography;
        elastic_stiffness::SymmetricTensor{4}, 
        yield_limit,
        self_hardening, cross_hardening,
        t_star, n_exp)
    dyads = get_slip_dyads(crystal)
    N = get_num_slipsystems(crystal)
    DS = map(dyad -> dyad ⊡ elastic_stiffness, dyads)
    ES = SMatrix{N,N}((DS_i ⊡ dyad_j for DS_i in DS, dyad_j in dyads)...)
    HGND = (self_hardening - cross_hardening, cross_hardening)
    return GradCPlast(elastic_stiffness, ES, DS, HGND, yield_limit, t_star, n_exp)
end

# eps() required to avoid zero stiffness at τᵈⁱ = 0
overstress(m::GradCPlast, τᵈⁱ) = (sign(τᵈⁱ)/m.t_star) * ((abs(τᵈⁱ)/m.τy)^m.n_exp + eps())

function overstress_inverse(m::GradCPlast, Δγ)
    return sign(Δγ) * m.τy * (abs(Δγ)*m.t_star)^(1/m.n_exp)
end

slipnames(::GradCPlast{<:Any, 2}) = (:γ1, :γ2)
slipnames(::GradCPlast{<:Any,12}) = (:γ1, :γ2, :γ3, :γ4, :γ5, :γ6, :γ7, :γ8, :γ9, :γ10, :γ11, :γ12)
crssnames(::GradCPlast{<:Any, 2}) = (:τ1, :τ2)
crssnames(::GradCPlast{<:Any,12}) = (:τ1, :τ2, :τ3, :τ4, :τ5, :τ6, :τ7, :τ8, :τ9, :τ10, :τ11, :τ12)

struct F3Model{M}
    m::M
end

slipnames(m::F3Model) = slipnames(m.m)
crssnames(m::F3Model) = crssnames(m.m)

# For debugging
#= 
function FerriteAssembly.element_routine!(Ke, re, state, ae, m::F3Model{<:GradCPlast}, cv, buffer)
    FerriteAssembly.element_routine_ad!(Ke, re, state, ae, m, cv, buffer)
    if cellid(buffer) == 1
        println("Full Ke")
        display(Ke)
        parts = [:u => collect(dof_range(buffer, :u)), 
                 :γ => append!(Int[], (dof_range(buffer, n) for n in slipnames(m))...),
                 :τ => append!(Int[], (dof_range(buffer, n) for n in crssnames(m))...)]
        for (kn, k) in parts
            for (ln, l) in parts
                println("Ke[", kn, ", ", ln, "]")
                display(Ke[k, l])
            end
        end
    end
end
# =#

function FerriteAssembly.element_residual!(re, state, ae, mw::F3Model{<:GradCPlast{T,N,dim}}, cv, buffer) where {T,N,dim}
    m = mw.m
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
            Δγ = (γ[slipnr] - γold[slipnr])
            ϕ = overstress(m, τᵈⁱ[slipnr])
            τ_part = (Δγ - Δt * ϕ) * dΩ

            for (iτ, Iτ) in pairs(dof_range(buffer, crsskey))
                δNτ = shape_value(cv[:τ], q_point, iτ)
                re[Iτ] += δNτ * τ_part
            end
        end
    end
end

# TODO: Currently F2Model not tested (at all)
struct F2Model{M}
    m::M
end

slipnames(m::F2Model) = slipnames(m.m)
crssnames(m::F2Model) = crssnames(m.m)

function FerriteAssembly.element_residual!(re, state, ae, mw::F2Model{<:GradCPlast{T,N,dim}}, cv, buffer) where {T,N,dim}
    m = mw.m
    slipkeys = slipnames(m)
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
        
        τᵈⁱ = map(γ, γold) do (γi, γoldi)
            Δt * overstress_inverse(m, γi - γoldi)
        end

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
    end
end
