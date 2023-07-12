using ModelingToolkit, OrdinaryDiffEq, Plots
using StructuralIdentifiability, Optimization
using ForwardDiff, OptimizationNLopt, GlobalSensitivity
using OptimizationOptimJL

@parameters a b c d
@variables t N(t) P(t) Z(t)
D = Differential(t)

eqs =  [D(N) ~ a*P + b*Z - c*N*P,
        D(P) ~ c*N*P - d*P*Z - a*P,
        D(Z) ~ d*Z*P - b*Z]

@named sys = ODESystem(eqs)
u0 = [N => 5.0, P => 0.1, Z => 0.1]
p = [a => 0.05, b => 1.0, c => 25.003, d => 1.8]

tspan = (0.0, 1.0)

function create_sol(p)
    prob = ODEProblem(sys, u0, tspan, p, jac = true)
    sol = solve(prob, Rosenbrock23(), saveat = 0.1)
end

sol = create_sol(p)
plot(sol)

#structural identifiability
identifiability = assess_identifiability(sys, measured_quantities = [N, P, Z])

#practical identifiability
sensitivity = gsa(p -> mean(Array(create_sol(p)), dims = 2), eFAST(), [(0, 1.0), (0.5, 5.0), (10.0, 30.0), (1.0, 5.0)], samples=1000)
println(sensitivity.S1)
println(sensitivity.ST)

#reparameterize to eliminate `a`
eqs = [D(N) ~ P + b * Z - c * N * P,
    D(P) ~ c * N * P - d * P * Z - P,
    D(Z) ~ d * Z * P - b * Z]

@named sys = ODESystem(eqs)
p = [b => 1.0 / 0.05, c => 25.003 / 0.05, d => 1.8 / 0.05]
prob = ODEProblem(sys, u0, tspan .* 0.05, p, jac=true)
sol = solve(prob, Rosenbrock23(), saveat=0.1 .* 0.05)
plot(sol)

#data for fitting
data = Array(sol) .+ 0.1*randn(3, length(sol.t))
plot!(sol.t, data')

function obj(p, _ = nothing)
    global prob, data
    prob = remake(prob, p = p)
    sol = solve(prob, Rosenbrock23(), saveat = 0.005)
    return sum(abs2, Array(sol) .- data)
end

optf = OptimizationFunction(obj, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optf, zeros(3))
optres = solve(optprob, NLopt.LD_LBFGS())

prob = remake(prob, p = optres.minimizer)
sol = solve(prob, Rosenbrock23(), saveat = 0.005)
plot!(sol)

#easy interface for creating objective function
using DiffEqParamEstim

cost_func = build_loss_objective(prob, Rosenbrock23(), L2Loss(sol.t, data), Optimization.AutoForwardDiff())
optprob = OptimizationProblem(cost_func, zeros(3))
optres = solve(optprob, NLopt.LD_LBFGS())

prob = remake(prob, p=optres.minimizer)
sol = solve(prob, Rosenbrock23(), saveat=0.005)
plot!(sol)
