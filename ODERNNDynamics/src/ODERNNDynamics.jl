module ODERNNDynamics



using Conda
using DifferentialEquations
using Flux
using Flux.Optimise
using Parameters
using PyCall
using SciMLSensitivity
import StatsBase
using Zygote
using RLTypes
using OneHotArrays


export modelEnv, setReward, ODE_RNN, CombinedModel, train_step!, accuracy



function setReward(state_size, action_size)

    return Chain(Dense(state_size + action_size + state_size, 64, tanh),
                    Dense(64, 64, tanh),
                    Dense(64, 1))
                    
end


# struct
mutable struct ODE_RNN
    cell::Flux.RNNCell
    # hidden::Int
    f_theta::Chain
end


Flux.@functor ODE_RNN

#outer constructor

function ODE_RNN(input_size::Int, hidden_size::Int)
    cell = Flux.RNNCell(input_size, hidden_size)
    f_theta = Chain(Dense(hidden_size, 32, tanh), Dense(32, hidden_size))#, Dense(100,hidden_size, tanh))
    ODE_RNN(cell, f_theta)
end



function (m::ODE_RNN)(timestamps, datapoints, hidden)

    h = Zygote.Buffer(hidden)

    for (i, el) in enumerate(timestamps)

        tspan = (el - 1.f0, el)
        f!(u, p, t) = m.f_theta(u)
        prob = ODEProblem(f!, h[:,i], tspan)
        # sol = solve(prob, AutoTsit5(Rodas4()), reltol=1e-3, abstol=1e-5, save_everystep = false)#, Tsit5(), reltol=1e-8, abstol=1e-8)
        sol = solve(prob, alg_hints = [:stiff], save_everystep = false, save_start = false)#, Tsit5(), reltol=1e-8, abstol=1e-8)
        # if sol.retcode != :Success
        #     println("It happens here")
        # end
        h[:,i+1] = m.cell(sol, datapoints[:,i])[1]

    end

    
    return copy(h[:,2:end])

end


mutable struct CombinedModel
    m::ODE_RNN
    d::Chain
end

Flux.@functor CombinedModel


# function CombinedModel(model, decoder)
#     CombinedModel(model, decoder)
# end



function (m::CombinedModel)(timestamps, datapoints, hidden)
    
    z = m.m(timestamps, datapoints, hidden)

    return m.d(z)

end




function accuracy(y_true, y_pred, tolerance)
    correct = sum(abs.(y_true .- y_pred) .<= tolerance)
    return correct / length(y_true)
end



function train_step!(environment::ContinuousEnvironment, S, A, R, S´, T, hidden, timestamps, fθ, model_opt)
# function train_step!(S, A, R, S´, T, hidden, timestamps, fθ, Rϕ, model_opt, reward_opt)

    sum(T[1:(end-1)]) > 0 && return
    X = vcat(S, A)



    dθ = Flux.gradient(m -> Flux.Losses.mse(m(timestamps, X, hidden), S´), fθ)
    Flux.update!(model_opt, fθ, dθ[1])
    
    # dϕ = Flux.gradient(m -> Flux.Losses.mse(m(vcat(S, A, S´)), hcat(R...)), Rϕ)
    # Flux.update!(reward_opt, Rϕ, dϕ[1])


end

function train_step!(environment::DiscreteEnvironment, S, A, R, S´, T, hidden, timestamps, fθ, model_opt, ep::EnvParameter)
# function train_step!(S, A, R, S´, T, hidden, timestamps, fθ, Rϕ, model_opt, reward_opt)


    sum(T[1:(end-1)]) > 0 && return
    X = vcat(S, onehotbatch(vcat(A...), ep.labels))


    dθ = Flux.gradient(m -> Flux.Losses.mse(m(timestamps, X, hidden), S´), fθ)
    Flux.update!(model_opt, fθ, dθ[1])
    
    # dϕ = Flux.gradient(m -> Flux.Losses.mse(m(vcat(S, A, S´)), hcat(R...)), Rϕ)
    # Flux.update!(reward_opt, Rϕ, dϕ[1])


end



function modelEnv(environment::ContinuousEnvironment, modelParams::ModelParameter)

    gym = pyimport("gymnasium")
    
    if environment isa LunarLanderContinuous
        global env = gym.make("LunarLander-v2", continuous=true)
    elseif environment isa BipedalWalker
        global env = gym.make("BipedalWalker-v3")
    elseif environment isa Pendulum
        global env = gym.make("Pendulum-v1")
    else
        println("Environment not supported")
    end

    envParams = EnvParameter()

    # Reset Parameters
    ## ActionenvP
    envParams.action_size =        env.action_space.shape[1]
    envParams.action_bound =       env.action_space.high[1]
    envParams.action_bound_high =  env.action_space.high
    envParams.action_bound_low =   env.action_space.low
    ## States
    envParams.state_size =         env.observation_space.shape[1]
    envParams.state_bound_high =   env.observation_space.high
    envParams.state_bound_low =    env.observation_space.low


    model = ODE_RNN(envParams.state_size + envParams.action_size, modelParams.hidden)
    decoder = Chain(Dense(modelParams.hidden, envParams.state_size))

    fθ = CombinedModel(model, decoder)
    model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)

    # Rϕ = setReward(envParams.state_size, envParams.action_size)
    # reward_opt = Flux.setup(Flux.Optimise.Adam(modelParams.reward_η), Rϕ)

    train_buffer = ReplayBuffer(modelParams.buffer_size)
    test_buffer = ReplayBuffer(modelParams.buffer_size)

    episode = 0

    collectTransitions!(train_buffer, env, modelParams.collect_train)
    collectTransitions!(test_buffer, env, modelParams.collect_test)



    for j in 1:modelParams.training_episodes
    
    
        for k in 1:modelParams.batch_size
        # Threads.@threads for k in 1:modelParams.batch_size
        
            #modelParams.trajectory = StatsBase.sample(1:8)
        
            timestamps = Float32[i for i in 1:modelParams.trajectory]
            hidden = zeros32(modelParams.hidden, size(timestamps)[1] + 1)
        
            S, A, R, S´, T = sample(train_buffer, ModelMethod(), modelParams.trajectory)
            train_step!(environment, S, A, R, S´, T, hidden, timestamps, fθ, model_opt)
        
        end
    

        losses = []
        #reward_losses = []
        acc = []

        for l in 1:5
            
            S, A, R, S´, T = sample(train_buffer, ModelMethod(), modelParams.trajectory)
            sum(T[1:(end-1)]) > 0 && continue
            X = vcat(S, A)
            timestamps = Float32[i for i in 1:size(X)[2]]
            hidden = zeros32(modelParams.hidden, size(timestamps)[1]+1)
            Ŝ = fθ(timestamps, X, hidden)

            push!(losses, Flux.Losses.mse(Ŝ, S´))
        #    push!(reward_losses, Flux.Losses.mse(Rϕ(vcat(S, A, Ŝ)), hcat(R...)))
            push!(acc, accuracy(S´, Ŝ, modelParams.tolerance))

        end
        
        push!(modelParams.model_loss, StatsBase.mean(losses))
        #push!(modelParams.reward_loss, StatsBase.mean(reward_losses))
        push!(modelParams.train_acc, StatsBase.mean(acc))

        acc = []
        for m in 1:5

            S, A, R, S´, T = sample(test_buffer, ModelMethod(), modelParams.trajectory)
            sum(T[1:(end-1)]) > 0 && continue
            X = vcat(S, A)
            timestamps = Float32[i for i in 1:size(X)[2]]
            hidden = zeros32(modelParams.hidden, size(timestamps)[1] + 1)
            Ŝ = fθ(timestamps, X, hidden)

            push!(acc, accuracy(S´, Ŝ, modelParams.tolerance))
        end

        push!(modelParams.test_acc, StatsBase.mean(acc))
        

        if j % modelParams.store_frequency == 0
            
            # S, A, R, S´, T = sample(test_buffer, ModelMethod(), modelParams.trajectory)
            # sum(T[1:(end-1)]) > 0 && continue
            # X = vcat(S, A)
            # timestamps = Float32[i for i in 1:size(X)[2]]
            # hidden = zeros32(modelParams.hidden, size(timestamps)[1] + 1)
            # Ŝ = fθ(timestamps, X, hidden)
            
            # @info "Current State: " S
            # @info "Action: " A 
            # @info "Next State: " S´
            # @info "Predicted Next State" Ŝ
            # @info "Difference" (S´ .- Ŝ)
            # # @info "Reward: " R'
            # # @info "Predicted Reward: " Rϕ(vcat(S, A, Ŝ))
            # sleep(2.)
            
            push!(modelParams.trained_model, deepcopy(fθ))
            #push!(modelParams.trained_reward, deepcopy(Rϕ))
        end
        
        println("Episode: $j | Train Accuracy: $(round(modelParams.train_acc[end], digits=2)) | Test Accuracy: $(round(modelParams.test_acc[end], digits=2)) | Model Loss: $(modelParams.model_loss[end])")# | Reward Loss: $(modelParams.reward_loss[end])")


    end

        
    return modelParams, model, decoder
    
end


# discrete
function modelEnv(environment::DiscreteEnvironment, modelParams::ModelParameter)

    gym = pyimport("gymnasium")
    
    if environment isa LunarLanderDiscrete
        global env = gym.make("LunarLander-v2")
    elseif environment isa Acrobot
        global env = gym.make("Acrobot-v1")
    else
        println("Environment not supported")
    end

    global envParams = EnvParameter()

    # Reset Parameters
    ## ActionenvP
    envParams.action_size =        environment isa Acrobot ? 3 : 4
    envParams.action_range =       0:envParams.action_size - 1
    envParams.labels =             collect(0:envParams.action_size - 1)
    # envParams.action_bound_high =  env.action_space.high
    # envParams.action_bound =       env.action_space.high[1]
    # envParams.action_bound_low =   env.action_space.low
    ## States
    envParams.state_size =         env.observation_space.shape[1]
    envParams.state_bound_high =   env.observation_space.high
    envParams.state_bound_low =    env.observation_space.low



    model = ODE_RNN(envParams.state_size + envParams.action_size, modelParams.hidden)
    decoder = Chain(Dense(modelParams.hidden, envParams.state_size))

    fθ = CombinedModel(model, decoder)
    model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)

    # Rϕ = setReward(envParams.state_size, envParams.action_size)
    # reward_opt = Flux.setup(Flux.Optimise.Adam(modelParams.reward_η), Rϕ)

    train_buffer = DiscreteReplayBuffer(modelParams.buffer_size)
    test_buffer = DiscreteReplayBuffer(modelParams.buffer_size)

    episode = 0

    collectTransitions!(train_buffer, env, modelParams.collect_train)
    collectTransitions!(test_buffer, env, modelParams.collect_test)



    for j in 1:modelParams.training_episodes
    
    
        for k in 1:modelParams.batch_size
        # Threads.@threads for k in 1:modelParams.batch_size
        
            #modelParams.trajectory = StatsBase.sample(1:8)
        
            timestamps = Float32[i for i in 1:modelParams.trajectory]
            hidden = zeros32(modelParams.hidden, size(timestamps)[1] + 1)
        
            S, A, R, S´, T = sample(train_buffer, ModelMethod(), modelParams.trajectory)
            train_step!(environment, S, A, R, S´, T, hidden, timestamps, fθ, model_opt, envParams)
        
        end
    

        losses = []
        #reward_losses = []
        acc = []

        for l in 1:5
            
            S, A, R, S´, T = sample(train_buffer, ModelMethod(), modelParams.trajectory)
            sum(T[1:(end-1)]) > 0 && continue
            X = vcat(S, onehotbatch(vcat(A...), envParams.labels))
            timestamps = Float32[i for i in 1:size(X)[2]]
            hidden = zeros32(modelParams.hidden, size(timestamps)[1]+1)
            Ŝ = fθ(timestamps, X, hidden)

            push!(losses, Flux.Losses.mse(Ŝ, S´))
        #    push!(reward_losses, Flux.Losses.mse(Rϕ(vcat(S, A, Ŝ)), hcat(R...)))
            push!(acc, accuracy(S´, Ŝ, modelParams.tolerance))

        end
        
        push!(modelParams.model_loss, StatsBase.mean(losses))
        #push!(modelParams.reward_loss, StatsBase.mean(reward_losses))
        push!(modelParams.train_acc, StatsBase.mean(acc))

        acc = []
        for m in 1:5

            S, A, R, S´, T = sample(test_buffer, ModelMethod(), modelParams.trajectory)
            sum(T[1:(end-1)]) > 0 && continue
            X = vcat(S, onehotbatch(vcat(A...), envParams.labels))
            timestamps = Float32[i for i in 1:size(X)[2]]
            hidden = zeros32(modelParams.hidden, size(timestamps)[1] + 1)
            Ŝ = fθ(timestamps, X, hidden)

            push!(acc, accuracy(S´, Ŝ, modelParams.tolerance))
        end

        push!(modelParams.test_acc, StatsBase.mean(acc))
        

        if j % modelParams.store_frequency == 0
            
            # S, A, R, S´, T = sample(test_buffer, ModelMethod(), modelParams.trajectory)
            # sum(T[1:(end-1)]) > 0 && continue
            # X = vcat(S, A)
            # timestamps = Float32[i for i in 1:size(X)[2]]
            # hidden = zeros32(modelParams.hidden, size(timestamps)[1] + 1)
            # Ŝ = fθ(timestamps, X, hidden)
            
            # @info "Current State: " S
            # @info "Action: " A 
            # @info "Next State: " S´
            # @info "Predicted Next State" Ŝ
            # @info "Difference" (S´ .- Ŝ)
            # # @info "Reward: " R'
            # # @info "Predicted Reward: " Rϕ(vcat(S, A, Ŝ))
            # sleep(2.)
            
            push!(modelParams.trained_model, deepcopy(fθ))
            #push!(modelParams.trained_reward, deepcopy(Rϕ))
        end
        
        println("Episode: $j | Train Accuracy: $(round(modelParams.train_acc[end], digits=2)) | Test Accuracy: $(round(modelParams.test_acc[end], digits=2)) | Model Loss: $(modelParams.model_loss[end])")# | Reward Loss: $(modelParams.reward_loss[end])")


    end

        
    return modelParams, model, decoder
    
end


end # module ODERNNDynamics


# mE = modelEnv(Acrobot(), ModelParameter(collect_train=100, collect_test=10, training_episodes=50, batch_size=512))

# mE = modelEnv("Pendulum-v1", ModelParameter(collect_train=100, collect_test=10, training_episodes=140, batch_size=512, hidden=10, model_η=0.0005))
# mE = modelEnv("Pendulum-v1", ModelParameter(collect_train=100, collect_test=10, training_episodes=140, batch_size=512, hidden=10, model_η=0.0005))
# mE = modelEnv("LunarLander-v2", ModelParameter(collect_train=100, collect_test=4, training_episodes=50, batch_size=512, hidden=10, model_η=0.0005))