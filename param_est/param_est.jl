using AlgebraicPetri
using OrdinaryDiffEq
using DiffEqFlux, Flux
using Catalyst
using Plots
using JSON
using CSV
using DataFrames

####################
# Define epi model #
####################
SIR = LabelledReactionNet{Float64, Float64}([:S=>1000.0, :I=>0.0, :R=>0.0],
                                            (:inf=>0.0008)=>((:S,:I)=>(:I,:I)),
                                            (:rec=>0.1)=>(:I=>:R));
SVIIR = LabelledReactionNet{Float64, Float64}([:S=>1000.0, :V=>0.0, :I_U=>1.0, :I_V=>0.0, :R=>0.0],
                                            (Symbol("β_{UU}")=>0.0008)=>((:S,:I_U)=>(:I_U,:I_U)),
                                            (Symbol("β_{UV}")=>0.0008)=>((:S,:I_V)=>(:I_U,:I_V)),
                                            (Symbol("β_{VU}")=>0.0008)=>((:V,:I_U)=>(:I_V,:I_U)),
                                            (Symbol("β_{VV}")=>0.0008)=>((:V,:I_V)=>(:I_V,:I_V)),
                                            (:γ_U=>0.1)=>(:I_U=>:R),
                                            (:γ_V=>0.1)=>(:I_V=>:R),
                                            (:ν=>0.0)=>(:S=>:V));

# Initial parameter guesses. The first 2 arguments are the rate parameters (in
# order of definition) and the next 3 are state parameters
SIR_param_guess = [1e-8, 1e-4, 1e7, 1e5, 1e-10]
SVIIR_param_guess = [1e-8, 1e-8,1e-8, 1e-8,1/14, 1/14, 1e-4, 6e5, 1e1, 1e3, 1e-1, 1e-1]

# Choose model
model = SVIIR
param_guess = SVIIR_param_guess

# Calculating the instantaneous rate of infection in the population
calc_inf_sviir(s,p) = s[1]*s[3]*p[1] + s[1]*s[4]*p[2] + s[2]*s[3]*p[3] + s[2]*s[4]*p[4]
calc_inf_sir(s, p) = s[1]*s[2]*p[1]

# Define the loss function given rates and states
function make_loss(pn, prob, times, data; calc_inf=calc_inf_sir, states=Dict(), rates=Dict())
    function loss(p)
        cur_p = exp.(p)
        u0 = exp.(p[(1:ns(pn)) .+ nt(pn)])
        for (k,v) in rates
            cur_p[k] = v
        end
        for (k,v) in states
            u0[k] = v
        end
        prob′ = remake(prob, p=cur_p, u0=u0, tspan=(0.0,150.0))
        sol = solve(prob′, Tsit5())
        sum(abs2, data .- [calc_inf(sol(t), cur_p) for t in times]), sol
    end
end

# Create a Catalyst model
sviir_rxn = ReactionSystem(SVIIR)
prob = ODEProblem(sviir_rxn, concentrations(model), (0.0,150.0), vcat(rates(model), concentrations(model)))

# Ingest data
state_data = CSV.read("georgia.csv", DataFrame)

# Get the change in "reported case" numbers for each day
times = 1:(length(state_data[:, :date])-1)
data = state_data[2:end, :cases] .- state_data[1:(end-1), :cases];

# Set a range of time to average over (in days)
avg_range = 3
week_times = 1:(length(data) - avg_range * 2)
week_avg = map((avg_range+1):(length(data)-avg_range)) do i
    sum(data[(i-avg_range):(i+avg_range)]) ./ (2*avg_range+1)
end
plot(week_times, week_avg)
savefig("real_data.png")

# Generate loss function (this fixes values give in keyword `rates` and `states`).
# The fixed values included here are described in the caption of Figure 5
l_func = make_loss(model, prob, week_times, week_avg; calc_inf=calc_inf_sviir, 
                                                      rates=Dict(5=>1/14, 6=>1/14), 
                                                      states=Dict(2=>1e-9, 4=>1e-9));

# Fit parameters
p = DiffEqFlux.sciml_train(l_func,log.(param_guess),ADAM(0.1),maxiters = 1000)
@show p

# Plot fit data against actual data
n_solve = solve(remake(prob, p=exp.(p.u), u0=exp.(p.u[(1:ns(model)) .+ nt(model)])), Tsit5())
p_times = range(1,149, length=1000)
plot(week_times, week_avg; linewidth=4, label="Georgia Data", yrange=(0,10000), yaxis="New daily infections (persons)", xaxis="Time (days)")
plot!(p_times, [calc_inf_sviir(n_solve(t), exp.(p.u)) for t in p_times]; linewidth=4, label="Estimated Data")
savefig("fit_results.png")

# Plot simulation with population in each of the states
plot(n_solve; labels=reshape(String.(snames(model)), (1,ns(model))), linewidth=4, yaxis="Population (persons)", xaxis="Time (days)")
savefig("sim_results.png")