using GradientCrystalPlasticity
using Documenter

DocMeta.setdocmeta!(GradientCrystalPlasticity, :DocTestSetup, :(using GradientCrystalPlasticity); recursive=true)

makedocs(;
    modules=[GradientCrystalPlasticity],
    authors="Knut Andreas Meyer and contributors",
    sitename="GradientCrystalPlasticity.jl",
    format=Documenter.HTML(;
        canonical="https://KnutAM.github.io/GradientCrystalPlasticity.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/KnutAM/GradientCrystalPlasticity.jl",
    devbranch="main",
)
