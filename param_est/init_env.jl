using Pkg
Pkg.activate(".")
Pkg.add(PackageSpec(;name="Catalyst", version="10.0.0"))
Pkg.add(PackageSpec(;name="AlgebraicPetri", rev="Catalyst_v10"))
Pkg.instantiate()
