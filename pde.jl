using ModelingToolkit, MethodOfLines, NonlinearSolve, Plots
using Optimization, OptimizationPolyalgorithms, ComponentArrays
using ModelingToolkit: Interval
using Lux, NeuralPDE, Random

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
dx = 0.1
dy = 0.1

bcs = [u(0.0, y) ~ 0,
    u(1.0, y) ~ y,
    u(x, 0.0) ~ 0,
    u(x, 1.0) ~ x]


# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)]

@named pdesys = PDESystem([eq], bcs, domains, [x, y], [u(x, y)])

# Note that we pass in `nothing` for the time variable `t` here since we
# are creating a stationary problem without a dependence on time, only space.
discretization = MOLFiniteDifference([x => dx, y => dy], nothing, approx_order=2)

prob = discretize(pdesys, discretization)
sol = NonlinearSolve.solve(prob, NewtonRaphson())

xs = sol[x]
ys = sol[y]
u_sol = sol[u(x, y)]

p1 = plot(xs, ys, u_sol, linetype=:contourf, title="MOL Solution");

# Neural network
dim = 2 # number of dimensions
chain = Lux.Chain(Dense(dim, 16, Lux.σ), Dense(16, 16, Lux.σ), Dense(16, 1))

# Discretization
dx = 0.1
discretization = PhysicsInformedNN(chain, GridTraining(dx))

prob = discretize(pdesys, discretization)

#Callback function
callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, PolyOpt(), callback=callback, maxiters=500)
phi = discretization.phi

u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
    (length(xs), length(ys)))
u_predict[end,end] = 0.0

p2 = plot(xs, ys, u_predict, linetype=:contourf, title="NeuralPDE solution");

plot(p1, p2)
