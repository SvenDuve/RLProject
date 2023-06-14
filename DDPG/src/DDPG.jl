module DDPG

using Flux, Flux.Optimise
import Flux.params
import Distributions
using Statistics
using Conda
using PyCall
using Parameters 
using UnPack
using MLUtils
using RLTypes
# import Base.push!


export  agent,
        renderEnv, 
        setCritic, 
        setActor,
        train_step!


        
function setCritic(state_size, action_size)

    # working basic config
    # return Chain(Dense(state_size + action_size, 400, relu),
    #                 Dense(400, 300, relu),
    #                 Dense(300, 1))
    return Chain(Dense(state_size + action_size, 64, relu),
                    Dense(64, 64, relu),
                    Dense(64, 1))
                    
end


function setActor(state_size, action_size)

    return Chain(Dense(state_size, 64, relu),
                    Dense(64, 64, relu),
                    Dense(64, action_size, tanh))
    #working basci config                
    # return Chain(Dense(state_size, 400, relu),
    #                 Dense(400, 300, relu),
    #                 Dense(300, action_size, tanh))

end



# Define the action, actor_loss, and critic_loss functions
function action(model, state, train, ep, hp)
    if train
        a = model(state) .+ clamp.(rand(Distributions.Normal{Float32}(0.f0, hp.expl_noise), size(ep.action_size)), -hp.noise_clip, hp.noise_clip)
        return clamp.(a, ep.action_bound_low, ep.action_bound_high)
    else
        return model(state)
    end
end



function soft_update!(target_model, main_model, τ)
    for (target_param, main_param) in zip(Flux.params(target_model), Flux.params(main_model))
        target_param .= τ * main_param .+ (1 - τ) * target_param
    end
end



function verify_update(target_model, main_model)
    for (i, (target_param, main_param)) in enumerate(zip(Flux.params(target_model), Flux.params(main_model)))
        diff = main_param - target_param
        println("Difference for parameter $i:")
        println(diff)
    end
end



function train_step!(S, A, R, S´, T, μθ, μθ´, Qϕ, Qϕ´, opt_critic, opt_actor, ap::Parameter)

    Y = R' .+ ap.γ * (1 .- T)' .* Qϕ´(vcat(S´, μθ´(S´)))   

    # Gradient Descent Critic
    dϕ = Flux.gradient(m -> Flux.Losses.mse(m(vcat(S, A)), Y), Qϕ)
    Flux.update!(opt_critic, Qϕ, dϕ[1])
    
    # Gradient Descent Actor
    dθ = Flux.gradient(m -> -mean(Qϕ(vcat(S, m(S)))), μθ)
    Flux.update!(opt_actor, μθ, dθ[1])

    # Soft update
    soft_update!(Qϕ´, Qϕ, 0.005)
    soft_update!(μθ´, μθ, 0.005)

end


function agent(environment::ContinuousEnvironment, agentParams::AgentParameter)

    gym = pyimport("gymnasium")
    # gym = pyimport("gym")
    
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

    episode = 1

    μθ = setActor(envParams.state_size, envParams.action_size)
    μθ´= deepcopy(μθ)
    Qϕ = setCritic(envParams.state_size, envParams.action_size)
    Qϕ´= deepcopy(Qϕ)

    global opt_critic = Flux.setup(Flux.Optimise.Adam(agentParams.critic_η), Qϕ)
    global opt_actor = Flux.setup(Flux.Optimise.Adam(agentParams.actor_η), μθ)
    
    buffer = ReplayBuffer(agentParams.buffer_size)

    while episode ≤ agentParams.training_episodes

        frames = 0
        s, info = env.reset()
        episode_rewards = 0
        t = false
        
        for step in 1:agentParams.maximum_episode_length
            
            a = action(μθ, s, true, envParams, agentParams)
            s´, r, terminated, truncated, _ = env.step(a)

            terminated | truncated ? t = true : t = false

            episode_rewards += r
            
            remember(buffer, s, a, r, s´, t)

            if episode > agentParams.train_start

                S, A, R, S´, T = sample(buffer, AgentMethod(), agentParams.batch_size)
                train_step!(S, A, R, S´, T, μθ, μθ´, Qϕ, Qϕ´, opt_critic, opt_actor, agentParams)
                
            end

            
            s = s´
            frames += 1
            
            if t
                env.close()
                break
            end
            
        end

        
        if episode % agentParams.store_frequency == 0
            push!(agentParams.trained_agents, deepcopy(μθ))
        end

        actor_loss = []
        critic_loss = []

        for l in 1:10
            
            S, A, R, S´, T = sample(buffer, AgentMethod(), agentParams.batch_size)
            
            # actor loss 
            push!(actor_loss, -mean(Qϕ(vcat(S, μθ(S)))))
            
            # critic loss
            Y = R' .+ agentParams.γ * (1 .- T)' .* Qϕ´(vcat(S´, μθ´(S´)))
            push!(critic_loss, Flux.Losses.mse(Qϕ(vcat(S, A)), Y))

        end
        
        push!(agentParams.critic_loss, mean(critic_loss))
        push!(agentParams.actor_loss, mean(actor_loss))
        push!(agentParams.episode_steps, frames)
        push!(agentParams.episode_reward, episode_rewards)

        println("Episode: $episode | Cumulative Reward: $(round(episode_rewards, digits=2)) | Critic Loss: $(agentParams.critic_loss[end]) | Actor Loss: $(agentParams.actor_loss[end]) | Steps: $(frames)")
        
        episode += 1
    
    end
    
    return agentParams
    
end


# Works
# hp = agent("BipedalWalker-v3", HyperParameter(expl_noise=0.1f0, noise_clip=0.3f0, training_episodes=2000, maximum_episode_length=3000, train_start=20, batch_size=128, critic_η=0.0001, actor_η=0.0001))
# hp = agent("Pendulum-v1", HyperParameter(expl_noise=0.2f0, noise_clip=0.5f0, training_episodes=300, maximum_episode_length=3000, train_start=20, batch_size=128))
# hp = agent("LunarLander-v2", HyperParameter(expl_noise=0.2f0, training_episodes=1000, maximum_episode_length=3000, train_start=20, batch_size=128, critic_η=0.0001, actor_η=0.0001))


# hp = agent("MountainCarContinuous-v0", HyperParameter(expl_noise=0.2f0, noise_clip=0.5f0, training_episodes=300, maximum_episode_length=3000, train_start=20, batch_size=128))


# hp = agent(DDPG(), "BipedalWalker-v3", HyperParameter(expl_noise=0.2f0, noise_clip=0.5f0, training_episodes=200, maximum_episode_length=1000, train_start=20, batch_size=64, store_frequency=20))
# hp = agent("Pendulum-v1", HyperParameter(expl_noise=0.2f0, noise_clip=0.5f0, training_episodes=200, maximum_episode_length=1000, train_start=20, batch_size=64, store_frequency=20))

# Workds
# aP = agent(Pendulum(), AgentParameter(training_episodes=500))
# aP = agent(LunarLanderContinuous(), AgentParameter(training_episodes=500))



function renderEnv(environment::ContinuousEnvironment, policy, seed=42)

    gym = pyimport("gymnasium")
    
    if environment isa LunarLanderContinuous
        global env = gym.make("LunarLander-v2", continuous = true, render_mode="human")
    elseif environment isa BipedalWalker
        global env = gym.make("BipedalWalker-v3", render_mode="human")
    elseif environment isa Pendulum
        global env = gym.make("Pendulum-v1", render_mode="human")
    else
        println("Environment not supported")
    end

    s, info = env.reset(seed=seed)

    R = []
    notSolved = true

    while notSolved
        
        a = action(policy, s, false, EnvParameter(), AgentParameter()) 

        s´, r, terminated, truncated, _ = env.step(a)

        terminated | truncated ? t = true : t = false

        append!(R, r)

        sleep(0.05)
        s = s´
        notSolved = !t
    end

    println("The agent has achieved return $(sum(R))")

    env.close()

end #renderEnv


end # module DDPG
