module NNDynamics


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


export modelEnv, setReward, NNModel, train_step!, accuracy



mutable struct NNModel
    input::Chain
    hidden::Chain
    output::Chain
end

Flux.@functor NNModel


function NNModel(input_size::Int, ode_size::Int, output_size::Int)
    input = Chain(Dense(input_size, ode_size))
    hidden = Chain(Dense(ode_size, 32, tanh), Dense(32, ode_size))#, Dense(100,hidden_size, tanh))
    output = Chain(Dense(ode_size, output_size))
    NNModel(input, hidden, output)
end


function (m::NNModel)(datapoints)
    
    return m.output(m.hidden(m.input(datapoints)))

end

function accuracy(y_true, y_pred, tolerance)
    correct = sum(abs.(y_true .- y_pred) .<= tolerance)
    return correct / length(y_true)
end



#function train_step!(S, A, R, S´, T, fθ, Rϕ, model_opt, reward_opt)
function train_step!(environment::ContinuousEnvironment, S, A, R, S´, T, fθ, model_opt)

    X = vcat(S, A)

    # Train both critic networks

    sum(T[1:(end-1)]) > 0 && return

    dθ = Flux.gradient(m -> Flux.Losses.mse(m(X), S´), fθ)
    Flux.update!(model_opt, fθ, dθ[1])
    
    # dϕ = Flux.gradient(m -> Flux.Losses.mse(m(vcat(S, A, S´)), hcat(R...)), Rϕ)
    # Flux.update!(reward_opt, Rϕ, dϕ[1])


end


function train_step!(environment::DiscreteEnvironment, S, A, R, S´, T, fθ, model_opt, ep::EnvParameter)

    X = vcat(S, onehotbatch(vcat(A...), ep.labels))
    
    # Train both critic networks

    sum(T[1:(end-1)]) > 0 && return


    dθ = Flux.gradient(m -> Flux.Losses.mse(m(X), S´), fθ)
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


    global envParams = EnvParameter()

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

    fθ = NNModel(envParams.state_size + envParams.action_size, modelParams.ode_size, envParams.state_size)
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
        
            # timestamps = Float32[i for i in 1:modelParams.trajectory]
        
            S, A, R, S´, T = sample(train_buffer, ModelMethod(), modelParams.trajectory)
            train_step!(environment, S, A, R, S´, T, fθ, model_opt)
        
        end
    

        losses = []
        # reward_losses = []
        acc = []

        for l in 1:10
            
            S, A, R, S´, T = sample(train_buffer, ModelMethod(), modelParams.trajectory)
            sum(T[1:(end-1)]) > 0 && continue
            X = vcat(S, A)
            Ŝ = fθ(X)

            push!(losses, Flux.Losses.mse(Ŝ, S´))
            # push!(reward_losses, Flux.Losses.mse(Rϕ(vcat(S, A, Ŝ)), hcat(R...)))
            push!(acc, accuracy(S´, Ŝ, modelParams.tolerance))

        end
        
        push!(modelParams.model_loss, StatsBase.mean(losses))
        # push!(modelParams.reward_loss, StatsBase.mean(reward_losses))
        push!(modelParams.train_acc, StatsBase.mean(acc))

        acc = []
        for m in 1:10

            S, A, R, S´, T = sample(test_buffer, ModelMethod(), modelParams.trajectory)
            sum(T[1:(end-1)]) > 0 && continue
            
            X = vcat(S, A)
            Ŝ = fθ(X)

            push!(acc, accuracy(S´, Ŝ, modelParams.tolerance))
        end

        push!(modelParams.test_acc, StatsBase.mean(acc))
        

        if j % modelParams.store_frequency == 0
            
            # S, A, R, S´, T = sample(test_buffer, ModelMethod(), modelParams.trajectory)
            # sum(T[1:(end-1)]) > 0 && continue
            # X = vcat(S, A)
            # Ŝ = fθ(X)
            
            # @info "Current State: " S
            # @info "Action: " A 
            # @info "Next State: " S´
            # @info "Predicted Next State" Ŝ
            # @info "Difference" (S´ .- Ŝ)
            # # @info "Reward: " R'
            # # @info "Predicted Reward: " Rϕ(vcat(S, A, Ŝ))
            # sleep(1.)
            
            push!(modelParams.trained_model, deepcopy(fθ))
            # push!(modelParams.trained_reward, deepcopy(Rϕ))
        end
        
        println("Episode: $j | Train Accuracy: $(round(modelParams.train_acc[end], digits=2)) | Test Accuracy: $(round(modelParams.test_acc[end], digits=2)) | Model Loss: $(modelParams.model_loss[end])")# | Reward Loss: $(modelParams.reward_loss[end])")


    end

        
    return modelParams
    
end


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


    fθ = NNModel(envParams.state_size + envParams.action_size, modelParams.ode_size, envParams.state_size)
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
        
            # timestamps = Float32[i for i in 1:modelParams.trajectory]
        
            S, A, R, S´, T = sample(train_buffer, ModelMethod(), modelParams.trajectory)
            train_step!(environment, S, A, R, S´, T, fθ, model_opt, envParams)
        
        end
    

        losses = []
        # reward_losses = []
        acc = []

        for l in 1:10
            
            S, A, R, S´, T = sample(train_buffer, ModelMethod(), modelParams.trajectory)
            sum(T[1:(end-1)]) > 0 && continue

            X = vcat(S, onehotbatch(vcat(A...), envParams.labels))
            Ŝ = fθ(X)

            push!(losses, Flux.Losses.mse(Ŝ, S´))
            # push!(reward_losses, Flux.Losses.mse(Rϕ(vcat(S, A, Ŝ)), hcat(R...)))
            push!(acc, accuracy(S´, Ŝ, modelParams.tolerance))

        end
        
        push!(modelParams.model_loss, StatsBase.mean(losses))
        # push!(modelParams.reward_loss, StatsBase.mean(reward_losses))
        push!(modelParams.train_acc, StatsBase.mean(acc))

        acc = []
        for m in 1:10

            S, A, R, S´, T = sample(test_buffer, ModelMethod(), modelParams.trajectory)
            sum(T[1:(end-1)]) > 0 && continue

            X = vcat(S, onehotbatch(vcat(A...), envParams.labels))
            Ŝ = fθ(X)

            push!(acc, accuracy(S´, Ŝ, modelParams.tolerance))
        end

        push!(modelParams.test_acc, StatsBase.mean(acc))
        

        if j % modelParams.store_frequency == 0
            
            # S, A, R, S´, T = sample(test_buffer, ModelMethod(), modelParams.trajectory)
            # sum(T[1:(end-1)]) > 0 && continue
            # timestamps = Float32[i for i in 1:length(T)]
            # X = vcat(S, A)
            # Ŝ = fθ(timestamps, X)
            
            # @info "Current State: " S
            # @info "Action: " A 
            # @info "Next State: " S´
            # @info "Predicted Next State" Ŝ
            # @info "Difference" (S´ .- Ŝ)
            # # @info "Reward: " R'
            # # @info "Predicted Reward: " Rϕ(vcat(S, A, Ŝ))
            # sleep(1.)
            
            push!(modelParams.trained_model, deepcopy(fθ))
            # push!(modelParams.trained_reward, deepcopy(Rϕ))
        end
        
        println("Episode: $j | Train Accuracy: $(round(modelParams.train_acc[end], digits=2)) | Test Accuracy: $(round(modelParams.test_acc[end], digits=2)) | Model Loss: $(modelParams.model_loss[end])")# | Reward Loss: $(modelParams.reward_loss[end])")


    end

        
    return modelParams
    
end

# mP = modelEnv(Pendulum(), ModelParameter())

# mP = modelEnv(Pendulum(), ModelParameter(collect_train=100, collect_test=10, training_episodes=100, batch_size=512))
# mP = modelEnv(LunarLanderContinuous(), ModelParameter(collect_train=100, collect_test=10, training_episodes=100, batch_size=512))
# mP = modelEnv(BipedalWalker(), ModelParameter(collect_train=100, collect_test=10, training_episodes=100, batch_size=512))


end # module NNDynamics
