using Plots
using DataFrames
using Statistics
using Interpolations
using LaTeXStrings

using BSON
using Flux
using Flux: loadmodel!

using RLTypes
using ODERNNDynamics
using NNDynamics
using NODEDynamics

using Conda
using PyCall


using LinearAlgebra: norm

function mean_squared_error(y_true, y_pred)
    mse = mean(norm(y_true[i]-y_pred[i])^2 for i in 1:length(y_true))
    return mse
end



environment_name = "Pendulum-v1"
gym = pyimport("gymnasium")
env = gym.make(environment_name)


fθ_1 = NODE(4, 32, 3)
fθ_2 = NODE(4, 32, 3)
fθ_3 = NODE(4, 32, 3)
fθ_1 = loadmodel!(fθ_1, BSON.load("./MBRL/pendulum_node_model.bson")[:mE].trained_model[1])
fθ_2 = loadmodel!(fθ_2, BSON.load("./MBRL/pendulum_node_model.bson")[:mE].trained_model[10])
fθ_3 = loadmodel!(fθ_3, BSON.load("./MBRL/pendulum_node_model.bson")[:mE].trained_model[end])

model_1 = ODE_RNN(4, 10)
decoder_1 = Chain(Dense(10, 3))
ode_1 = CombinedModel(model_1, decoder_1)

model_2 = ODE_RNN(4, 10)
decoder_2 = Chain(Dense(10, 3))
ode_2 = CombinedModel(model_1, decoder_1)

model_3 = ODE_RNN(4, 10)
decoder_3 = Chain(Dense(10, 3))
ode_3 = CombinedModel(model_1, decoder_1)


ode_1 = loadmodel!(ode_1, BSON.load("./MBRL/pendulum_model.bson")[:mE][1].trained_model[1])
ode_2 = loadmodel!(ode_2, BSON.load("./MBRL/pendulum_model.bson")[:mE][1].trained_model[10])
ode_3 = loadmodel!(ode_3, BSON.load("./MBRL/pendulum_model.bson")[:mE][1].trained_model[end])



nnθ_1 = NNModel(4, 32, 3)
nnθ_2 = NNModel(4, 32, 3)
nnθ_3 = NNModel(4, 32, 3)
nnθ_1 = loadmodel!(nnθ_1, BSON.load("./MBRL/pendulum_nn_model.bson")[:pendulum_nn_model].trained_model[1])
nnθ_2 = loadmodel!(nnθ_2, BSON.load("./MBRL/pendulum_nn_model.bson")[:pendulum_nn_model].trained_model[10])
nnθ_3 = loadmodel!(nnθ_3, BSON.load("./MBRL/pendulum_nn_model.bson")[:pendulum_nn_model].trained_model[end])



theta_next = []
thetadot_next = []
torque_next = []

theta_pred_1 = []
thetadot_pred_1 = []
torque_pred_1 = []

theta_pred_2 = []
thetadot_pred_2 = []
torque_pred_2 = []

theta_pred_3 = []
thetadot_pred_3 = []
torque_pred_3 = []


nn_theta_pred_1 = []
nn_thetadot_pred_1 = []
nn_torque_pred_1 = []

nn_theta_pred_2 = []
nn_thetadot_pred_2 = []
nn_torque_pred_2 = []

nn_theta_pred_3 = []
nn_thetadot_pred_3 = []
nn_torque_pred_3 = []

s′_1 = []
s′_2 = []
s′_3 = []
s_hat_1 = []
s_hat_2 = []
s_hat_3 = []
nn_1 = []
nn_2 = []
nn_3 = []

state, info = env.reset()

for _ in 1:200

    a = env.action_space.sample()
    next_state, r, term, trunc, _ = env.step(a)
    next_state_pred_1 = fθ_1([1.f0], vcat(state, a))
    next_state_pred_2 = fθ_2([1.f0], vcat(state, a))
    next_state_pred_3 = fθ_3([1.f0], vcat(state, a))

    nn_next_state_pred_1 = nnθ_1(vcat(state, a))
    nn_next_state_pred_2 = nnθ_2(vcat(state, a))
    nn_next_state_pred_3 = nnθ_3(vcat(state, a))
    
    push!(s′_1, next_state)
    push!(s′_2, next_state)
    push!(s′_3, next_state)
    push!(s_hat_1, next_state_pred_1)
    push!(s_hat_2, next_state_pred_2)
    push!(s_hat_3, next_state_pred_3)
    push!(nn_1, nn_next_state_pred_1)
    push!(nn_2, nn_next_state_pred_2)
    push!(nn_3, nn_next_state_pred_3)
    
    push!(theta_next, next_state[1])
    push!(thetadot_next, next_state[2])
    push!(torque_next, next_state[3])
    
    
    push!(theta_pred_1, next_state_pred_1[1])
    push!(thetadot_pred_1, next_state_pred_1[2])
    push!(torque_pred_1, next_state_pred_1[3])
    
    push!(theta_pred_2, next_state_pred_2[1])
    push!(thetadot_pred_2, next_state_pred_2[2])
    push!(torque_pred_2, next_state_pred_2[3])
    
    push!(theta_pred_3, next_state_pred_3[1])
    push!(thetadot_pred_3, next_state_pred_3[2])
    push!(torque_pred_3, next_state_pred_3[3])
    
    
    push!(nn_theta_pred_1, nn_next_state_pred_1[1])
    push!(nn_thetadot_pred_1, nn_next_state_pred_1[2])
    push!(nn_torque_pred_1, nn_next_state_pred_1[3])
    
    push!(nn_theta_pred_2, nn_next_state_pred_2[1])
    push!(nn_thetadot_pred_2, nn_next_state_pred_2[2])
    push!(nn_torque_pred_2, nn_next_state_pred_2[3])
    
    push!(nn_theta_pred_3, nn_next_state_pred_3[1])
    push!(nn_thetadot_pred_3, nn_next_state_pred_3[2])
    push!(nn_torque_pred_3, nn_next_state_pred_3[3])
    
    state = next_state
    
    term | trunc ? break : continue
    
end


from = 1
to = 50

p1 = scatter(theta_next[from:to], thetadot_next[from:to], torque_next[from:to], mc=:blue, ms=1.5, ma=0.5, label=L"(s, a) \rightarrow s\prime", xlabel=L"\cos{\theta}", ylabel=L"\sin{\theta}", zlabel=L"\tau")
p1 = scatter!(theta_pred_1[from:to], thetadot_pred_1[from:to], torque_pred_1[from:to], mc=:red, ms=1.5, ma=0.5, label=L"\hat{f}: (s, a) \rightarrow \hat{s}")

p2 = scatter(theta_next[from:to], thetadot_next[from:to], torque_next[from:to], mc=:blue, ma=0.5, ms=1.5, label=L"(s, a) \rightarrow s\prime", xlabel=L"\cos{\theta}", ylabel=L"\sin{\theta}", zlabel=L"\tau")
p2 = scatter!(theta_pred_2[from:to], thetadot_pred_2[from:to], torque_pred_2[from:to], mc=:red, ma=0.5, ms=1.5, label=L"\hat{f}: (s, a) \rightarrow \hat{s}")

p3 = scatter(theta_next[from:to], thetadot_next[from:to], torque_next[from:to], mc=:blue, ma=0.5,ms=1.5, label=L"(s, a) \rightarrow s\prime", xlabel=L"\cos{\theta}", ylabel=L"\sin{\theta}", zlabel=L"\tau")
p3 = scatter!(theta_pred_3[from:to], thetadot_pred_3[from:to], torque_pred_3[from:to], mc=:red, ma=0.5, ms=1.5, label=L"\hat{f}: (s, a) \rightarrow \hat{s}")





plot(p1, p2, p3, layout=grid(1, 3), 
size=(1800, 500), 
title=["n = 5" "n = 50" "n=500"],
titlefont = font(14),
legend=:topleft)

savefig("output/NODE_predictions.png")


p4 = scatter(theta_next[from:to], thetadot_next[from:to], torque_next[from:to], mc=:blue, ms=1.5, ma=0.5, label=L"(s, a) \rightarrow s\prime", xlabel=L"\theta", ylabel=L"\dot{\theta}", zlabel=L"\tau")
p4 = scatter!(nn_theta_pred_1[from:to], nn_thetadot_pred_1[from:to], nn_torque_pred_1[from:to], mc=:red, ms=1.5, ma=0.5, label=L"\hat{f}: (s, a) \rightarrow \hat{s}")

p5 = scatter(theta_next[from:to], thetadot_next[from:to], torque_next[from:to], mc=:blue, ma=0.5, ms=1.5, label=L"(s, a) \rightarrow s\prime", xlabel=L"\theta", ylabel=L"\dot{\theta}", zlabel=L"\tau")
p5 = scatter!(nn_theta_pred_2[from:to], nn_thetadot_pred_2[from:to], nn_torque_pred_2[from:to], mc=:red, ma=0.5, ms=1.5, label=L"\hat{f}: (s, a) \rightarrow \hat{s}")

p6 = scatter(theta_next[from:to], thetadot_next[from:to], torque_next[from:to], mc=:blue, ma=0.5,ms=1.5, label=L"(s, a) \rightarrow s\prime", xlabel=L"\theta", ylabel=L"\dot{\theta}", zlabel=L"\tau")
p6 = scatter!(nn_theta_pred_3[from:to], nn_thetadot_pred_3[from:to], nn_torque_pred_3[from:to], mc=:red, ma=0.5, ms=1.5, label=L"\hat{f}: (s, a) \rightarrow \hat{s}")





plot(p4, p5, p6, layout=grid(1, 3), 
size=(1800, 500), 
title=["n = 5" "n = 50" "n=500"],
titlefont = font(14),
legend=:topleft)

savefig("output/NODE_predictions.png")





s′_1 = []
s′_2 = []
s′_3 = []
s_hat_1 = []
s_hat_2 = []
s_hat_3 = []
# ode_hat_1 = []
# ode_hat_2 = []
# ode_hat_3 = []
nn_1 = []
nn_2 = []
nn_3 = []

state, info = env.reset()
# z_1 = zeros32(10, 2)
# z_2 = zeros32(10, 2)
# z_3 = zeros32(10, 2)
# t_step = [1.f0]

for _ in 1:100000
    
    a = env.action_space.sample()
    next_state, r, term, trunc, _ = env.step(a)
    next_state_pred_1 = fθ_1([1.f0], vcat(state, a))
    next_state_pred_2 = fθ_2([1.f0], vcat(state, a))
    next_state_pred_3 = fθ_3([1.f0], vcat(state, a))

    # latent_1 = model_1(t_step, vcat(state, a), z_1)
    # ode_next_state_pred_1 = decoder_1(latent_1)
    
    # latent_2 = model_2(t_step, vcat(state, a), z_2)
    # ode_next_state_pred_2 = decoder_2(latent_2)
    
    # latent_3 = model_3(t_step, vcat(state, a), z_3)
    # ode_next_state_pred_3 = decoder_3(latent_3)
    
    nn_next_state_pred_1 = nnθ_1(vcat(state, a))
    nn_next_state_pred_2 = nnθ_2(vcat(state, a))
    nn_next_state_pred_3 = nnθ_3(vcat(state, a))
    
    push!(s′_1, next_state)
    push!(s′_2, next_state)
    push!(s′_3, next_state)
    push!(s_hat_1, next_state_pred_1)
    push!(s_hat_2, next_state_pred_2)
    push!(s_hat_3, next_state_pred_3)
    # push!(ode_hat_1, ode_next_state_pred_1)
    # push!(ode_hat_2, ode_next_state_pred_2)
    # push!(ode_hat_3, ode_next_state_pred_3)
    push!(nn_1, nn_next_state_pred_1)
    push!(nn_2, nn_next_state_pred_2)
    push!(nn_3, nn_next_state_pred_3)
    
    
    state = next_state
    # z_1[:,1] = latent_1
    # z_2[:,1] = latent_2
    # z_3[:,1] = latent_3
    
    if term | trunc
        state, info = env.reset()
    end
    
end


mean_squared_error(s′_1, s_hat_1)
mean_squared_error(s′_1, ode_hat_1)
mean_squared_error(s′_1, nn_1)

mean_squared_error(s′_2, s_hat_2)
mean_squared_error(s′_2, ode_hat_2)
mean_squared_error(s′_2, nn_2)

mean_squared_error(s′_3, s_hat_3)
mean_squared_error(s′_3, ode_hat_3)
mean_squared_error(s′_3, nn_3)  