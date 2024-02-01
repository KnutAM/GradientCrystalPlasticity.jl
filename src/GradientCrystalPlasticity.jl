module GradientCrystalPlasticity
using Tensors, Ferrite, StaticArrays
using FerriteAssembly

include("utils.jl")
include("stardot.jl")
include("SlipSystems.jl")
include("GradCPlast.jl")

end
