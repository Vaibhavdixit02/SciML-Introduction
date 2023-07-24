using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets

@parameters x y t
@variables u(..) v(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

∇²(u) = Dxx(u) + Dyy(u)

brusselator_f(x, y, t) = (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

α = 10.

u0(x,y,t) = 22(y*(1-y))^(3/2)
v0(x,y,t) = 27(x*(1-x))^(3/2)

eq = [Dt(u(x,y,t)) ~ 1. + v(x,y,t)*u(x,y,t)^2 - 4.4*u(x,y,t) + α*∇²(u(x,y,t)) + brusselator_f(x, y, t),
       Dt(v(x,y,t)) ~ 3.4*u(x,y,t) - v(x,y,t)*u(x,y,t)^2 + α*∇²(v(x,y,t))]

domains = [x ∈ Interval(x_min, x_max),
              y ∈ Interval(y_min, y_max),
              t ∈ Interval(t_min, t_max)]

# Periodic BCs
bcs = [u(x,y,0) ~ u0(x,y,0),
       u(0,y,t) ~ u(1,y,t),
       u(x,0,t) ~ u(x,1,t),

       v(x,y,0) ~ v0(x,y,0),
       v(0,y,t) ~ v(1,y,t),
       v(x,0,t) ~ v(x,1,t)] 

@named pdesys = PDESystem(eq,bcs,domains,[x,y,t],[u(x,y,t),v(x,y,t)])
N = 32

order = 2 # This may be increased to improve accuracy of some schemes

# Integers for x and y are interpreted as number of points. Use a Float to directtly specify stepsizes dx and dy.
discretization = MOLFiniteDifference([x=>N, y=>N], t, approx_order=order)

@time prob = discretize(pdesys, discretization)
@time sol = solve(prob, KenCarp47(), saveat=0.1)

using LinearAlgebra, LinearSolve, Krylov 
@time sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()), saveat = 0.1)


using Symbolics
du0 = copy(prob.u0)
@time jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> prob.f(du, u, prob.p, 0.0), du0, prob.u0)

f = ODEFunction(prob.f.f; jac_prototype = float.(jac_sparsity))
sparse_prob = ODEProblem(f, prob.u0, prob.tspan, prob.p)
@btime sol = solve(sparse_prob, KenCarp47(linsolve=KrylovJL_GMRES()), saveat = 0.1)

@btime sol = solve(sparse_prob, KenCarp47(linsolve=UMFPACKFactorization()), saveat = 0.1)

using IncompleteLU
function incompletelu(W, du, u, p, t, newW, Plprev, Prprev, solverdata)
  if newW === nothing || newW
      Pl = ilu(convert(AbstractMatrix, W), τ = 50.0)
  else
      Pl = Plprev
  end
  Pl, nothing
end
@btime sol = solve(sparse_prob, KenCarp47(linsolve = KrylovJL_GMRES(), precs = incompletelu, concrete_jac = true), saveat = 0.1)
