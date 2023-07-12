module DQN

using RLTypes
using Flux
import Distributions
using Statistics
using Conda
using PyCall
using Parameters
using UnPack
using MLUtils
using OneHotArrays
using Rewards



export greet, agent, set_q_function, ϵ_greedy, train_step!, set_greediness!


greet() = print("Hello World!")


function set_q_function(state_size, action_size) 

    return Chain(Dense(state_size, 64, relu),
                    Dense(64, 64, relu),
                    Dense(64, action_size))

end

function set_greediness!(aP::AgentParameter, step_counter::Int64) 

    # random exploration before train_start or random number less than ϵ
    if step_counter < aP.random_exploration
        aP.ϵ = 1.0
    else
        aP.ϵ -= (aP.ϵ_max - aP.ϵ_min) / aP.ϵ_greedy_reduction
        # aP.ϵ -= exp(-(aP.ϵ_greedy_reduction / frames))
        aP.ϵ = max(aP.ϵ, aP.ϵ_min)
    end

end

function ϵ_greedy(Qϕ, s, aP::AgentParameter, eP::EnvParameter)

    if rand() < aP.ϵ    
        return rand(eP.labels)
    else
        return argmax(Qϕ(s)) - 1
    end

end


function train_step!(S, A, R, S´, T, Qϕ, Qϕ´, opt_critic, ap::Parameter, ep::EnvParameter)
    
    
    Y = R .+ ap.γ .* (1 .- T) .* maximum(Qϕ´(S´), dims=1)'

    dϕ = Flux.gradient(m -> Flux.Losses.huber_loss(sum(onehotbatch(vcat(A...), ep.labels) .* m(vcat(S)), dims=1), Y'), Qϕ)
    Flux.update!(opt_critic, Qϕ, dϕ[1])


end



function agent(environment::DiscreteEnvironment, agentParams::AgentParameter) 

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

    
    global Qϕ = set_q_function(envParams.state_size, envParams.action_size)
    global Qϕ´ = deepcopy(Qϕ)
    
    global opt_critic = Flux.setup(Flux.Optimise.Adam(agentParams.critic_η), Qϕ)
    
    buffer = DiscreteReplayBuffer(agentParams.buffer_size)
    
    
    episode = 1
    frames = 1
    steps = 0
    s, info = env.reset()
    episode_rewards = 0
    t = false

    while true

        if agentParams.train_type isa Epoch
            steps > agentParams.training_epochs * agentParams.epoch_length - 1 ? break : nothing
        elseif agentParams.train_type isa Episode
            episode > agentParams.training_episodes ? break : nothing
        end

        set_greediness!(agentParams, steps)

        a = ϵ_greedy(Qϕ, s, agentParams, envParams)
        s´, r, terminated, truncated, _ =  env.step(a)
        steps += 1
        
        terminated | truncated ? t = true : t = false
        
        episode_rewards += r
        append!(agentParams.all_rewards, r)

        remember(buffer, s, a, r, s´, t)
        
        
        if episode > agentParams.train_start
            S, A, R, S´, T = sample(buffer, AgentMethod(), agentParams.batch_size)
            train_step!(S, A, R, S´, T, Qϕ, Qϕ´, opt_critic, agentParams, envParams)
        end

        if episode % 2 == 0
            Qϕ´ = deepcopy(Qϕ)
        end

        
        s = s´
        frames += 1

        
        
        if t
            s, info = env.reset()
            t = false
            critic_loss = []
            
            for l in 1:10
                
                S, A, R, S´, T = sample(buffer, AgentMethod(), agentParams.batch_size)
                
                # critic loss
                Y = R .+ agentParams.γ .* (1 .- T) .* maximum(Qϕ´(S´), dims=1)'
                push!(critic_loss, Flux.Losses.huber_loss(sum(onehotbatch(vcat(A...), envParams.labels) .* Qϕ(vcat(S)), dims=1), Y'))
                
            end
            
            if episode % agentParams.store_frequency == 0
                push!(agentParams.trained_agents, deepcopy(Qϕ))
            end
            
            push!(agentParams.critic_loss, mean(critic_loss))
            push!(agentParams.episode_steps, frames)
            push!(agentParams.episode_reward, episode_rewards)
            
            println("Episode: $episode | Cumulative Reward: $(round(episode_rewards, digits=2)) | Critic Loss: $(agentParams.critic_loss[end])") #  | Actor Loss: $(agentParams.actor_loss[end]) | Steps: $(frames)")
            
            frames = 0
            episode += 1
            episode_rewards = 0
            
            continue
            
        end
        
        
        

    end

    return agentParams

end


end # module DQN
