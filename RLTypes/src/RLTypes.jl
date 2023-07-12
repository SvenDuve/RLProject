module RLTypes

using Parameters
import StatsBase


export Parameter,
        EnvParameter,
        AgentParameter, 
        ModelParameter,
        Buffer,
        ReplayBuffer,
        DiscreteReplayBuffer,
        Methods,
        AgentMethod,
        ModelMethod,
        Episode,
        Epoch,
        Environments,
        DiscreteEnvironment,
        ContinuousEnvironment,
        Acrobot,
        LunarLanderDiscrete,
        Pendulum,
        LunarLanderContinuous,
        BipedalWalker,
        Models,
        NODEModel,
        ODERNNModel,
        remember,
        sample,
        collectTransitions!


abstract type Parameter end
abstract type Buffer end
abstract type Methods end
abstract type Environments end
abstract type Models end

abstract type DiscreteEnvironment <: Environments end
abstract type ContinuousEnvironment <: Environments end

struct Acrobot <: DiscreteEnvironment end
struct LunarLanderDiscrete <: DiscreteEnvironment end

struct Pendulum <: ContinuousEnvironment end
struct LunarLanderContinuous <: ContinuousEnvironment end
struct BipedalWalker <: ContinuousEnvironment end

struct AgentMethod <: Methods end
struct ModelMethod <: Methods end
struct Episode <: Methods end
struct Epoch <: Methods end

struct NODEModel <: Models end
struct ODERNNModel <: Models end




@with_kw mutable struct EnvParameter <: Parameter
    # Dimensions
    ## Actions
    action_size::Int =                      1
    action_bound::Float32 =                 1.f0
    action_bound_high::Array{Float32} =     [1.f0]
    action_bound_low::Array{Float32} =      [-1.f0]
    ## Discrete Parameters 
    action_range::UnitRange{Int64}=         1:2
    labels::Array{Int64} =                   [1, 2]

    ## States
    state_size::Int =                       1
    state_bound_high::Array{Float32} =      [1.f0]
    state_bound_low::Array{Float32} =       [1.f0]
end

@with_kw mutable struct AgentParameter <: Parameter
    train_type::Methods =                   Episode()
    # Buffer size
    buffer_size::Int =                      1000000
    # Exploration Continuous
    expl_noise::Float32 =                   0.2f0
    noise_clip::Float32 =                   1.f0
    # Exploratin Discrete
    random_exploration::Int =               1000
    ϵ_greedy_reduction::Float32 =           10000.f0
    ϵ::Float32 =                            1.f0
    ϵ_min::Float32 =                        0.1f0
    ϵ_max::Float32 =                        1.f0
    ϵ_decay::Float32 =                      0.999f0
    # Training Metrics
    training_episodes::Int =                20
    training_epochs::Int =                  10
    epoch_length::Int =                     2000
    maximum_episode_length::Int =           2000
    train_start:: Int =                     10
    batch_size::Int =                       64
    # Metrics
    episode_reward::Array{Float32} =        []
    all_rewards::Array{Float32} =           []
    critic_loss::Array{Float32} =           [0.f0]
    actor_loss::Array{Float32} =            [0.f0]
    episode_steps::Array{Int} =             []
    # Discount
    γ::Float32 =                            0.99f0
    # Learning Rates
    critic_η::Float64 =                     0.001
    actor_η::Float64 =                      0.001
    # Agents
    store_frequency::Int =                  100
    trained_agents =                        []
end

@with_kw mutable struct ModelParameter <: Parameter
    train::Bool =                           false
    # Buffer size
    buffer_size::Int =                      1000000
    collect_train::Int =                    5
    collect_test::Int =                     5
    # Exploration
    expl_noise::Float32 =                   0.2f0
    noise_clip::Float32 =                   1.f0
    # Training Metrics
    training_episodes::Int =                5
    maximum_episode_length::Int =           10
    train_start:: Int =                     1
    batch_size::Int =                       8
    trajectory::Int =                       8
    actionPlans::Int =                      10
    # Metrics
    episode_reward::Array{Float32} =        []
    model_loss::Array{Float32} =            []
    reward_loss::Array{Float32} =           []
    episode_steps::Array{Int} =             []
    # Discount
    γ::Float32 =                            0.99f0
    # Learning Rates
    model_η::Float64 =                      0.0005
    hidden::Int =                           10
    ode_size::Int =                         32
    reward_η::Float64 =                     0.0005
    # Agents
    store_frequency::Int =                  5
    retrain::Int =                          2000
    trained_model =                        []
    trained_reward =                        []
    # Accuracy
    train_acc =                              []
    test_acc =                              []
    tolerance::Float64 =                    0.05
end

# Define the experience replay buffer
mutable struct ReplayBuffer <: Buffer
    capacity::Int
    memory::Vector{Tuple{Vector{Float32}, Vector{Float32}, Float32, Vector{Float32}, Bool}}
    pos::Int
end

mutable struct DiscreteReplayBuffer <: Buffer
    capacity::Int
    memory::Vector{Tuple{Vector{Float32}, Int, Float32, Vector{Float32}, Bool}}
    pos::Int
end

# outer constructor for the Replay Buffer
function ReplayBuffer(capacity::Int)
    memory = []
    return ReplayBuffer(capacity, memory, 1)
end

function DiscreteReplayBuffer(capacity::Int)
    memory = []
    return DiscreteReplayBuffer(capacity, memory, 1)
end


function remember(buffer::Buffer, state, action, reward, next_state, done)
    if length(buffer.memory) < buffer.capacity
        push!(buffer.memory, (state, action, reward, next_state, done))
    else
        buffer.memory[buffer.pos] = (state, action, reward, next_state, done)
    end
    buffer.pos = mod1(buffer.pos + 1, buffer.capacity)
end

function sample(buffer::ReplayBuffer, method::AgentMethod, batch_size::Int)
    batch = rand(buffer.memory, batch_size)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for (s, a, r, ns, d) in batch
        push!(states, s)
        push!(actions, a)
        push!(rewards, r)
        push!(next_states, ns)
        push!(dones, d)
    end
    return hcat(states...), hcat(actions...), rewards, hcat(next_states...), dones
end

function sample(buffer::DiscreteReplayBuffer, method::AgentMethod, batch_size::Int)
    batch = rand(buffer.memory, batch_size)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for (s, a, r, ns, d) in batch
        push!(states, s)
        push!(actions, a)
        push!(rewards, r)
        push!(next_states, ns)
        push!(dones, d)
    end
    return hcat(states...), hcat(actions...), rewards, hcat(next_states...), dones
end



function sample(buffer::ReplayBuffer, method::ModelMethod, batch_size::Int)

    # State progression verified manually with small batch_size
    
    start = StatsBase.sample(1:(size(buffer.memory)[1] - batch_size))
    batch = buffer.memory[start:(start+batch_size-1)]
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for (s, a, r, ns, d) in batch
        push!(states, s)
        push!(actions, a)
        push!(rewards, r)
        push!(next_states, ns)
        push!(dones, d)
    end
    return hcat(states...), hcat(actions...), rewards, hcat(next_states...), dones
end


function sample(buffer::DiscreteReplayBuffer, method::ModelMethod, batch_size::Int)

    # State progression verified manually with small batch_size
    
    start = StatsBase.sample(1:(size(buffer.memory)[1] - batch_size))
    batch = buffer.memory[start:(start+batch_size-1)]
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for (s, a, r, ns, d) in batch
        push!(states, s)
        push!(actions, a)
        push!(rewards, r)
        push!(next_states, ns)
        push!(dones, d)
    end
    return hcat(states...), hcat(actions...), rewards, hcat(next_states...), dones
end


function collectTransitions!(buffer, env, n_transitions) 
    
    for i in 1:n_transitions
        
        frames = 0
        s, info = env.reset()
        episode_rewards = 0
        t = false
        
        while true
            
            #a = action(μθ, s, true, envParams, modelParams)
            a = env.action_space.sample()
            s´, r, terminated, truncated, _ = env.step(a)
            
            terminated | truncated ? t = true : t = false
            
            episode_rewards += r
            
            remember(buffer, s, a, r, s´, t)
            
            
            s = s´
            frames += 1
            
            if t
                env.close()
                break
            end
            
        end
        
    end
    
end
    
    
end # module RLTypes


