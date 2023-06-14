module MBRL

using DDPG: setActor, setCritic, action, train_step!, DDPG
# using ODERNNDynamics: modelEnv, setReward, ODE_RNN, CombinedModel, train_step!, accuracy, ODERNNDynamics
using ODERNNDynamics: ODE_RNN, CombinedModel#, train_step!, accuracy, ODERNNDynamics
import ODERNNDynamics
using NODEDynamics: NODE 
import NODEDynamics

import DQN 

# using DDPG: HyperParameter as ddpp_hp
# using DyModelNODE: HyperParameter as dyModel_hp
import Distributions
using Statistics
import StatsBase
using Conda
using PyCall
using Parameters
using Rewards
using UnPack
using MLUtils
using Flux
using Flux: loadmodel!
using RLTypes
using BSON
using OneHotArrays



export MBRLAgent, ReplayBuffer, AgentParameter, ModelParameter



function MBRLAgent(model::ODERNNModel, environment::ContinuousEnvironment, agentParams::AgentParameter, modelParams::ModelParameter) 

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




    
    μθ = setActor(envParams.state_size, envParams.action_size)
    μθ´= deepcopy(μθ)
    Qϕ = setCritic(envParams.state_size, envParams.action_size)
    Qϕ´= deepcopy(Qϕ)
    global fθ
    global Rϕ

    
    
    
    
    opt_critic = Flux.setup(Flux.Optimise.Adam(agentParams.critic_η), Qϕ)
    opt_actor = Flux.setup(Flux.Optimise.Adam(agentParams.actor_η), μθ)
    
    agentBuffer = ReplayBuffer(agentParams.buffer_size)
    
    
    if modelParams.train
        
        println("Training Model...")
        learnedModel, model, decoder = ODERNNDynamics.modelEnv(environment, modelParams)
        fθ = deepcopy(learnedModel.trained_model[end])
        
        model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        
    else
        println("Loading Model...")        
        model = ODE_RNN(envParams.state_size + envParams.action_size, modelParams.hidden)
        decoder = Chain(Dense(modelParams.hidden, envParams.state_size))
        fθ = CombinedModel(model, decoder)

        if environment isa Pendulum
            fθ = loadmodel!(fθ, BSON.load("./MBRL/pendulum_model.bson")[:mE][1].trained_model[end])
            model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        elseif environment isa LunarLanderContinuous
            fθ = loadmodel!(fθ, BSON.load("./MBRL/lunarlander_model.bson")[:mE][1].trained_model[end])
            model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        elseif environment isa BipedalWalker
            println("We are struggeling as you know.")
            return
        end
        
        
    end
    
    
    
    println("Training Agent...")
    episode = 1
    global retrain = 1
    
    while episode ≤ agentParams.training_episodes
        
        frames = 1
        s, info = env.reset()
        episode_rewards = 0
        Rϕ = Reward(environment)
        Rϕ.prev_shaping = shaping(environment, s)   
        t = false
        z = zeros32(modelParams.hidden, 2)
        
        while true
            
            states = Dict([(k, zeros(Float32, (envParams.state_size, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            latent = Dict([(k, zeros(Float32, (modelParams.hidden, 2))) for k in 1:modelParams.actionPlans])
            actions = Dict([(k, zeros(Float32, (envParams.action_size, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            rewards = Dict([(k, zeros(Float32, (1, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            
            for k in 1:modelParams.actionPlans
                
                t_steps, states[k][:,1], latent[k][:,1:2] = 1.f0, s, z
                Rϕ´ = Reward(environment)
                Rϕ´.prev_shaping = Rϕ.prev_shaping
                
                for h in 2:modelParams.trajectory #Planning Horizon

                    actions[k][:,h-1] = action(μθ, states[k][:,h-1], true, envParams, agentParams)
                    latent[k][:,1] = model([t_steps], vcat(states[k][:,h-1], actions[k][:,h-1]), latent[k])
                    states[k][:,h] = decoder(latent[k][:,1])
                    rewards[k][1,h-1] = Rϕ´(environment, states[k][:,h-1], actions[k][:,h-1], states[k][:,h])

                end

                actions[k][:,modelParams.trajectory] = action(μθ, states[k][:,modelParams.trajectory], false, envParams, agentParams)
                
            end

            k_optimal = argmax([sum([modelParams.γ^step * rewards[k][step] for step in 1:(modelParams.trajectory -1)]) + modelParams.γ^8 * Qϕ(vcat(states[k][:,modelParams.trajectory], actions[k][:,modelParams.trajectory]))[1] for k in 1:modelParams.actionPlans])
            a = actions[k_optimal][:,1]

            s´, r, terminated, truncated, _ = env.step(a)
            terminated | truncated ? t = true : t = false       
            
             
            z´ = model([1.f0], vcat(s, a), z)
                        
            remember(agentBuffer, s, a, r, s´, t)
            
            episode_rewards += r

            
            if t
                env.close()
                break
            end
            
            if episode > agentParams.train_start   
                S, A, R, S´, T = sample(agentBuffer, AgentMethod(), agentParams.batch_size)
                DDPG.train_step!(S, A, R, S´, T, μθ, μθ´, Qϕ, Qϕ´, opt_critic, opt_actor, agentParams)
            end

            s, z[:,1] = s´, z´
            frames += 1
            
        end


        actor_loss = []
        critic_loss = []

        for l in 1:10
            
            S, A, R, S´, T = sample(agentBuffer, AgentMethod(), agentParams.batch_size)
            
            
            # actor loss 
            push!(actor_loss, -mean(Qϕ(vcat(S, μθ(S)))))
            
            # critic loss
            Y = R' .+ agentParams.γ * (1 .- T)' .* Qϕ´(vcat(S´, μθ´(S´)))
            push!(critic_loss, Flux.Losses.mse(Qϕ(vcat(S, A)), Y))

        end
        
        push!(agentParams.critic_loss, StatsBase.mean(critic_loss))
        push!(agentParams.actor_loss, StatsBase.mean(actor_loss))


        if episode % agentParams.store_frequency == 0
            push!(agentParams.trained_agents, deepcopy(μθ))
        end

        push!(agentParams.episode_steps, frames)
        push!(agentParams.episode_reward, episode_rewards)
        println("Episode: $episode | Cumulative Reward: $(round(episode_rewards, digits=2)) | Critic Loss: $(agentParams.critic_loss[end]) | Actor Loss: $(agentParams.actor_loss[end]) | Steps: $(frames)")
        
        # Retraining the model from the actual buffer
        if (sum(agentParams.episode_steps) ÷ (modelParams.retrain * retrain)) > 0
            println("Retraining Model...")

            for i in 1:modelParams.batch_size

                S, A, R, S´, T = sample(agentBuffer, ModelMethod(), modelParams.trajectory)

                timestamps = Float32[i for i in 1:size(S)[2]]
                hidden = zeros32(modelParams.hidden, size(timestamps)[1] + 1)

                ODERNNDynamics.train_step!(environment, S, A, R, S´, T, hidden, timestamps, fθ, model_opt)

            end
            println("Done Retraining Model...")

            global retrain += 1

        end

        
        episode += 1
        
    end

    return (agentParams, modelParams)
    
end


function MBRLAgent(model::NODEModel, environment::ContinuousEnvironment, agentParams::AgentParameter, modelParams::ModelParameter) 

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




    
    μθ = setActor(envParams.state_size, envParams.action_size)
    μθ´= deepcopy(μθ)
    Qϕ = setCritic(envParams.state_size, envParams.action_size)
    Qϕ´= deepcopy(Qϕ)
    global fθ
    global Rϕ

    
    
    
    
    opt_critic = Flux.setup(Flux.Optimise.Adam(agentParams.critic_η), Qϕ)
    opt_actor = Flux.setup(Flux.Optimise.Adam(agentParams.actor_η), μθ)
    
    agentBuffer = ReplayBuffer(agentParams.buffer_size)

    
    if modelParams.train
        
        println("Training Model...")
        learnedModel = NODEDynamics.modelEnv(environment, modelParams)
        fθ = deepcopy(learnedModel.trained_model[end])
        
        model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        
    else
        println("Loading Model...")        
        fθ = NODE(envParams.state_size + envParams.action_size, modelParams.ode_size, envParams.state_size)

        if environment isa Pendulum
            fθ = loadmodel!(fθ, BSON.load("./MBRL/pendulum_node_model.bson")[:mE].trained_model[end])
            model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        elseif environment isa LunarLanderContinuous
            fθ = loadmodel!(fθ, BSON.load("./MBRL/lunarlander_node_model.bson")[:lunarlander_node_model].trained_model[end])
            model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        elseif environment isa BipedalWalker
            println("We are struggeling as you know.")
            return
        end
        
        
    end
    
    
    
    println("Training Agent...")
    episode = 1
    global retrain = 1
    
    while episode ≤ agentParams.training_episodes
        
        frames = 1
        s, info = env.reset()
        episode_rewards = 0
        Rϕ = Reward(environment)
        Rϕ.prev_shaping = shaping(environment, s)   
        t = false
        
        while true
            

            states = Dict([(k, zeros(Float32, (envParams.state_size, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            actions = Dict([(k, zeros(Float32, (envParams.action_size, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            rewards = Dict([(k, zeros(Float32, (1, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            
            for k in 1:modelParams.actionPlans
                
                t_steps, states[k][:,1] = 1.f0, s
                Rϕ´ = Reward(environment)
                Rϕ´.prev_shaping = Rϕ.prev_shaping

                for h in 2:modelParams.trajectory #Planning Horizon

                    actions[k][:,h-1] = action(μθ, states[k][:,h-1], true, envParams, agentParams)
                    states[k][:,h] = fθ([t_steps], vcat(states[k][:,h-1], actions[k][:,h-1]))
                    rewards[k][1,h-1] = Rϕ´(environment, states[k][:,h-1], actions[k][:,h-1], states[k][:,h])

                end

                actions[k][:,modelParams.trajectory] = action(μθ, states[k][:,modelParams.trajectory], false, envParams, agentParams)

            end


            k_optimal = argmax([sum([modelParams.γ^step * rewards[k][step] for step in 1:(modelParams.trajectory -1)]) + modelParams.γ^8 * Qϕ(vcat(states[k][:,modelParams.trajectory], actions[k][:,modelParams.trajectory]))[1] for k in 1:modelParams.actionPlans])
            a = actions[k_optimal][:,1]

            s´, r, terminated, truncated, _ = env.step(a)
            terminated | truncated ? t = true : t = false       
                                    
            remember(agentBuffer, s, a, r, s´, t)
            
            episode_rewards += r

            
            if t
                env.close()
                break
            end
            
            if episode > agentParams.train_start   
                S, A, R, S´, T = sample(agentBuffer, AgentMethod(), agentParams.batch_size)
                DDPG.train_step!(S, A, R, S´, T, μθ, μθ´, Qϕ, Qϕ´, opt_critic, opt_actor, agentParams)
            end

            s = s´
            frames += 1
            
        end


        actor_loss = []
        critic_loss = []

        for l in 1:10
            
            S, A, R, S´, T = sample(agentBuffer, AgentMethod(), agentParams.batch_size)
            
            
            # actor loss 
            push!(actor_loss, -mean(Qϕ(vcat(S, μθ(S)))))
            
            # critic loss
            Y = R' .+ agentParams.γ * (1 .- T)' .* Qϕ´(vcat(S´, μθ´(S´)))
            push!(critic_loss, Flux.Losses.mse(Qϕ(vcat(S, A)), Y))

        end
        
        push!(agentParams.critic_loss, StatsBase.mean(critic_loss))
        push!(agentParams.actor_loss, StatsBase.mean(actor_loss))


        if episode % agentParams.store_frequency == 0
            push!(agentParams.trained_agents, deepcopy(μθ))
        end

        push!(agentParams.episode_steps, frames)
        push!(agentParams.episode_reward, episode_rewards)
        println("Episode: $episode | Cumulative Reward: $(round(episode_rewards, digits=2)) | Critic Loss: $(agentParams.critic_loss[end]) | Actor Loss: $(agentParams.actor_loss[end]) | Steps: $(frames)")
        
        # Retraining the model from the actual buffer
        if (sum(agentParams.episode_steps) ÷ (modelParams.retrain * retrain)) > 0
            println("Retraining Model...")

            for i in 1:modelParams.batch_size

                
                S, A, R, S´, T = sample(agentBuffer, ModelMethod(), modelParams.trajectory)
                #sum(T[1:(end-1)]) > 0 && continue
                NODEDynamics.train_step!(environment, S, A, R, S´, T, fθ, model_opt)

            end
            println("Done Retraining Model...")

            global retrain += 1

        end

        
        episode += 1
        
    end

    return (agentParams, modelParams)
    
end


function MBRLAgent(model::NODEModel, environment::DiscreteEnvironment, agentParams::AgentParameter, modelParams::ModelParameter) 

    gym = pyimport("gymnasium")

    if environment isa Acrobot 
        global env = gym.make("Acrobot-v1")
    elseif environment isa LunarLanderDiscrete
        global env = gym.make("LunarLander-v2")
    else
        println("Environment not supported")
    end


    global envParams = EnvParameter()

    # Reset Parameters
    envParams.action_size =        environment isa Acrobot ? 3 : 4
    envParams.action_range =       0:envParams.action_size - 1
    envParams.labels =             collect(0:envParams.action_size - 1)

    ## States
    envParams.state_size =         env.observation_space.shape[1]
    envParams.state_bound_high =   env.observation_space.high
    envParams.state_bound_low =    env.observation_space.low


    global Qϕ = DQN.set_q_function(envParams.state_size, envParams.action_size)
    global Qϕ´ = deepcopy(Qϕ)

    opt_critic = Flux.setup(Flux.Optimise.Adam(agentParams.critic_η), Qϕ)

        
    agentBuffer = DiscreteReplayBuffer(agentParams.buffer_size)
    
    
    if modelParams.train
        
        println("Training Model...")
        learnedModel = NODEDynamics.modelEnv(environment, modelParams)
        fθ = deepcopy(learnedModel.trained_model[end])
        
        model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        
    else
        println("Loading Model...")        
        fθ = NODE(envParams.state_size + envParams.action_size, modelParams.ode_size, envParams.state_size)
        if environment isa LunarLanderDiscrete
            fθ = loadmodel!(fθ, BSON.load("./MBRL/lunarlanderdiscrete_node_model.bson")[:lunarlanderdiscrete_node_model].trained_model[end])
            model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        elseif environment isa Acrobot
            fθ = loadmodel!(fθ, BSON.load("./MBRL/acrobot_node_model.bson")[:acrobot_node_model].trained_model[end])
            model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        else
            println("We are struggeling as you know.")
            return
        end
                
    end
    
    
    
    println("Training Agent...")
    episode = 1
    step_counter = 0
    global retrain = 1
    
    while episode ≤ agentParams.training_episodes
        
        frames = 0
        s, info = env.reset()
        episode_rewards = 0
        Rϕ = Reward(environment)
        Rϕ.prev_shaping = shaping(environment, s)   
        t = false
        
        while true

            DQN.set_greediness!(agentParams, step_counter)
            

            states = Dict([(k, zeros(Float32, (envParams.state_size, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            actions = Dict([(k, zeros(Float32, (1, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            rewards = Dict([(k, zeros(Float32, (1, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            
            for k in 1:modelParams.actionPlans
                
                t_steps, states[k][:,1] = 1.f0, s
                Rϕ´ = Reward(environment)
                Rϕ´.prev_shaping = Rϕ.prev_shaping

                for h in 2:modelParams.trajectory #Planning Horizon

                    actions[k][1,h-1] = DQN.ϵ_greedy(Qϕ, states[k][:,h-1], agentParams, envParams)
                    states[k][:,h] = fθ([t_steps], vcat(states[k][:,h-1], onehotbatch(vcat(actions[k][:,h-1]...), envParams.labels)))
                    rewards[k][1,h-1] = Rϕ´(environment, states[k][:,h-1], actions[k][1,h-1], states[k][:,h])
                    

                end
                
                actions[k][1,modelParams.trajectory] = DQN.ϵ_greedy(Qϕ, states[k][:,modelParams.trajectory], agentParams, envParams)

            end
    
            k_optimal = argmax([sum([modelParams.γ^step * rewards[k][step] for step in 1:(modelParams.trajectory -1)]) + modelParams.γ^modelParams.trajectory * sum(Qϕ(vcat(states[k][:,modelParams.trajectory])) .* onehotbatch(vcat(actions[k][:,modelParams.trajectory]...), envParams.labels)) for k in 1:modelParams.actionPlans])
            a = Int(actions[k_optimal][1])

            s´, r, terminated, truncated, _ = env.step(a)
            step_counter += 1

            terminated | truncated ? t = true : t = false       
            
            episode_rewards += r
                                    
            remember(agentBuffer, s, a, r, s´, t)
            
            
            if episode > agentParams.train_start   
                S, A, R, S´, T = sample(agentBuffer, AgentMethod(), agentParams.batch_size)
                DQN.train_step!(S, A, R, S´, T, Qϕ, Qϕ´, opt_critic, agentParams, envParams)
            end
                        
            s = s´
            frames += 1
            
            if t
                env.close()
                break
            end

        end

        critic_loss = []

        for l in 1:10
            
            S, A, R, S´, T = sample(agentBuffer, AgentMethod(), agentParams.batch_size)
            
            # critic loss
            Y = R .+ agentParams.γ .* (1 .- T) .* maximum(Qϕ´(S´), dims=1)'

            push!(critic_loss, Flux.Losses.huber_loss(sum(onehotbatch(vcat(A...), envParams.labels) .* Qϕ(vcat(S)), dims=1), Y'))

        end
        
        push!(agentParams.critic_loss, StatsBase.mean(critic_loss))

        if episode % 2 == 0
            Qϕ´ = deepcopy(Qϕ)
        end

        if episode % agentParams.store_frequency == 0
            push!(agentParams.trained_agents, deepcopy(Qϕ))
        end

        push!(agentParams.episode_steps, frames)
        push!(agentParams.episode_reward, episode_rewards)
        println("Episode: $episode | Cumulative Reward: $(round(episode_rewards, digits=2)) | Critic Loss: $(agentParams.critic_loss[end]) | Steps: $(frames)")
        
        # Retraining the model from the actual buffer
        if (sum(agentParams.episode_steps) ÷ (modelParams.retrain * retrain)) > 0
            println("Retraining Model...")

            for i in 1:modelParams.batch_size

                
                S, A, R, S´, T = sample(agentBuffer, ModelMethod(), modelParams.trajectory)
                #sum(T[1:(end-1)]) > 0 && continue
                NODEDynamics.train_step!(environment, S, A, R, S´, T, fθ, model_opt, envParams)

            end
            println("Done Retraining Model...")

            global retrain += 1

        end

        
        episode += 1
        
    end

    return (agentParams, modelParams)
    
end


function MBRLAgent(model::ODERNNModel, environment::DiscreteEnvironment, agentParams::AgentParameter, modelParams::ModelParameter) 

    gym = pyimport("gymnasium")

    if environment isa Acrobot 
        global env = gym.make("Acrobot-v1")
    elseif environment isa LunarLanderDiscrete
        global env = gym.make("LunarLander-v2")
    else
        println("Environment not supported")
    end


    global envParams = EnvParameter()

    # Reset Parameters

    envParams.action_size =        environment isa Acrobot ? 3 : 4
    envParams.action_range =       0:envParams.action_size - 1
    envParams.labels =             collect(0:envParams.action_size - 1)

    ## States
    envParams.state_size =         env.observation_space.shape[1]
    envParams.state_bound_high =   env.observation_space.high
    envParams.state_bound_low =    env.observation_space.low


    global Qϕ = DQN.set_q_function(envParams.state_size, envParams.action_size)
    global Qϕ´ = deepcopy(Qϕ)

    opt_critic = Flux.setup(Flux.Optimise.Adam(agentParams.critic_η), Qϕ)
    
    agentBuffer = DiscreteReplayBuffer(agentParams.buffer_size)

    if modelParams.train
        
        println("Training Model...")
        learnedModel, model, decoder = ODERNNDynamics.modelEnv(environment, modelParams)
        fθ = deepcopy(learnedModel.trained_model[end])
        
        model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        
    else

        println("Loading Model...")        
        model = ODE_RNN(envParams.state_size + envParams.action_size, modelParams.hidden)
        decoder = Chain(Dense(modelParams.hidden, envParams.state_size))
        fθ = CombinedModel(model, decoder)


        if environment isa LunarLanderDiscrete
            fθ = loadmodel!(fθ, BSON.load("./MBRL/lunarlanderdiscrete_odernn_model.bson")[:lunarlanderdiscrete_odernn_model][1].trained_model[end])
            model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        elseif environment isa Acrobot
            fθ = loadmodel!(fθ, BSON.load("./MBRL/acrobot_odernn_model.bson")[:acrobot_odernn_model][1].trained_model[end])
            model_opt = Flux.setup(Flux.Optimise.Adam(modelParams.model_η), fθ)
        else
            println("We are struggeling as you know.")
            return
        end
        
        
    end
    
    
    
    println("Training Agent...")
    episode = 1
    step_counter = 0
    global retrain = 1
    
    while episode ≤ agentParams.training_episodes
        
        frames = 0
        s, info = env.reset()
        episode_rewards = 0
        Rϕ = Reward(environment)
        Rϕ.prev_shaping = shaping(environment, s)   
        t = false
        z = zeros32(modelParams.hidden, 2)
        
        while true

            DQN.set_greediness!(agentParams, step_counter)
            

            states = Dict([(k, zeros(Float32, (envParams.state_size, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            latent = Dict([(k, zeros(Float32, (modelParams.hidden, 2))) for k in 1:modelParams.actionPlans])
            actions = Dict([(k, zeros(Float32, (1, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            rewards = Dict([(k, zeros(Float32, (1, modelParams.trajectory))) for k in 1:modelParams.actionPlans])
            
            for k in 1:modelParams.actionPlans
                
                t_steps, states[k][:,1], latent[k][:,1:2] = 1.f0, s, z
                Rϕ´ = Reward(environment)
                Rϕ´.prev_shaping = Rϕ.prev_shaping

                for h in 2:modelParams.trajectory #Planning Horizon


                    actions[k][1,h-1] = DQN.ϵ_greedy(Qϕ, states[k][:,h-1], agentParams, envParams)
                    latent[k][:,1] = model([t_steps], vcat(states[k][:,h-1], onehotbatch(actions[k][h-1], envParams.labels)), latent[k])
                    states[k][:,h] = decoder(latent[k][:,1])
                    rewards[k][1,h-1] = Rϕ´(environment, states[k][:,h-1], actions[k][1,h-1], states[k][:,h])
                    

                end
                
                actions[k][1,modelParams.trajectory] = DQN.ϵ_greedy(Qϕ, states[k][:,modelParams.trajectory], agentParams, envParams)

            end

            
            k_optimal = argmax([sum([modelParams.γ^step * rewards[k][step] for step in 1:(modelParams.trajectory -1)]) + modelParams.γ^modelParams.trajectory * sum(Qϕ(vcat(states[k][:,modelParams.trajectory])) .* onehotbatch(vcat(actions[k][:,modelParams.trajectory]...), envParams.labels)) for k in 1:modelParams.actionPlans])
            a = Int(actions[k_optimal][1])

            s´, r, terminated, truncated, _ = env.step(a)
            step_counter += 1

            terminated | truncated ? t = true : t = false       

            z´ = model([1.f0], vcat(s, onehotbatch(a, envParams.labels)), z)
            
                                    
            remember(agentBuffer, s, a, r, s´, t)
            episode_rewards += r
            
            
            if episode > agentParams.train_start   
                S, A, R, S´, T = sample(agentBuffer, AgentMethod(), agentParams.batch_size)
                DQN.train_step!(S, A, R, S´, T, Qϕ, Qϕ´, opt_critic, agentParams, envParams)
            end
                        
            s, z[:,1] = s´, z´
            frames += 1
            
            if t
                env.close()
                break
            end

        end


        critic_loss = []

        for l in 1:10
            
            S, A, R, S´, T = sample(agentBuffer, AgentMethod(), agentParams.batch_size)
            
            # critic loss
            Y = R .+ agentParams.γ .* (1 .- T) .* maximum(Qϕ´(S´), dims=1)'

            push!(critic_loss, Flux.Losses.huber_loss(sum(onehotbatch(vcat(A...), envParams.labels) .* Qϕ(vcat(S)), dims=1), Y'))

        end
        
        push!(agentParams.critic_loss, StatsBase.mean(critic_loss))

        if episode % 2 == 0
            Qϕ´ = deepcopy(Qϕ)
        end

        if episode % agentParams.store_frequency == 0
            push!(agentParams.trained_agents, deepcopy(Qϕ))
        end

        push!(agentParams.episode_steps, frames)
        push!(agentParams.episode_reward, episode_rewards)
        println("Episode: $episode | Cumulative Reward: $(round(episode_rewards, digits=2)) | Critic Loss: $(agentParams.critic_loss[end]) | Steps: $(frames)")
        
        # Retraining the model from the actual buffer
        if (sum(agentParams.episode_steps) ÷ (modelParams.retrain * retrain)) > 0
            println("Retraining Model...")

            for i in 1:modelParams.batch_size

                
                S, A, R, S´, T = sample(agentBuffer, ModelMethod(), modelParams.trajectory)

                timestamps = Float32[i for i in 1:size(S)[2]]
                hidden = zeros32(modelParams.hidden, size(timestamps)[1] + 1)
                ODERNNDynamics.train_step!(environment, S, A, R, S´, T, hidden, timestamps, fθ, model_opt, envParams)

            end
            println("Done Retraining Model...")

            global retrain += 1

        end

        
        episode += 1
        
    end

     return (agentParams, modelParams)
    
end

#MBRLAgent("Pendulum-v1", AgentParameter(training_episodes=5), ModelParameter(training_episodes=2, trajectory=5, hidden=10))
#MBRLAgent("Pendulum-v1", AgentParameter(training_episodes=100, train_start=5), ModelParameter(training_episodes=50, hidden=10))
# function checkComissar()
#     return DyModelNODE.modelEnv("Pendulum-v1", DyModelNODE.HyperParameter())
# end


end # module RNNMBRL

# This works for training the model
# mbP = MBRLAgent(Pendulum(), AgentParameter(training_episodes=10, train_start=2), ModelParameter(hidden=10, retrain = 1000, train=true))
# mbP = MBRLAgent(LunarLander(), AgentParameter(training_episodes=20, train_start=2), ModelParameter(hidden=10, retrain = 1000, train=true))
# mbP = MBRLAgent(BipedalWalker(), AgentParameter(training_episodes=20, train_start=2), ModelParameter(hidden=10, retrain = 1000, train=true))

# This works for using a pre-trained model
# mbP = MBRLAgent(Pendulum(), AgentParameter(training_episodes=10, train_start=2), ModelParameter(hidden=10, retrain = 1000))
# mbP = MBRLAgent(LunarLander(), AgentParameter(training_episodes=20, train_start=2), ModelParameter(hidden=10, retrain = 1000))
# mbP = MBRLAgent(BipedalWalker(), AgentParameter(training_episodes=20, train_start=2), ModelParameter(hidden=10, retrain = 1000, train=true))

# file naming: <environment>Models.bson