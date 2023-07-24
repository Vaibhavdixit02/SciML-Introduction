using DiffEqGPU, CUDA, OrdinaryDiffEq, StaticArrays

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func_ode = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func_ode, safetycopy = false)

@time sol = solve(monteprob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend(), 0.0),
    trajectories = 10_000, adaptive = false, dt = 0.1f0)

@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend(), 0.0),
    trajectories = 10_000, adaptive = false, dt = 0.1f0)


using Plots

plot(sol[1], idxs = (1,2,3))

plot(sol[2], idxs = (1,2,3))
