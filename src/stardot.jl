"""
    stardot(a::SVector, b::SVector)

Calculates the special contraction `∑ᵢ op(aᵢ, bᵢ)` where `op`
is determined by the element types of `a` and `b`. 
The following operations `op` are implemented
* If at least one element type `isa Number`: Standard multiplication, `*`
* If both `a` and `b` have `Vec` elements: Dot multiplication, `⋅`
* If both `a` and `b` are `SecondOrderTensors`: Double contraction, `⊡` (not implemented yet)

The operator `⋆` can be used directly, i.e. `a ⋆ b`
"""
function stardot end

const ⋆ = stardot # ⋆ has multiplicative precedence
# ref: https://github.com/JuliaLang/julia/blob/4b1bbeb2f22107d210bdd76d2a9b46ee59f5aaf5/src/julia-parser.scm#L24

# Number-Number
@inline function stardot(a::SVector{N,<:Number}, b::SVector{N,<:Number}) where {N}
    return a ⋅ b
end
# Number - Tensor
@inline function stardot(a::SVector{N,<:Number}, b::SVector{N,<:AbstractTensor}) where N
    return stardot(b, a)
end
function stardot(a::SVector{N,V}, b::SVector{N,<:Number}) where {N, V <: AbstractTensor}
    ret = zero(Tensors.get_base(V){promote_type(eltype(V), eltype(b))})
    @inbounds @simd for i = 1 : N
        ret += a[i] * b[i]
    end
    return ret
end
# Vec - Vec
function stardot(a::SVector{N,VA}, b::SVector{N,VB}) where {N, VA <: Vec{D}, VB <: Vec{D}} where D
    ret = zero(promote_type(eltype(VA), eltype(VB)))
    @inbounds @simd for i = 1 : N
        ret += a[i] ⋅ b[i]
    end
    return ret
end


