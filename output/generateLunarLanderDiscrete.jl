#julia -p 7 ./output/generateLunarLanderDiscrete.jl

using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin 
    using RLTypes
    using DQN
    using MBRL
    using BSON: @save 

    #exploreRate = [0.1, 0.2 ,0.3, 0.4]
end

# Epoch_DQN_MF_Agent_LunarLanderDiscrete -> Required
# Epoch_DQN_MBRL_NODE_LunarLanderDiscrete -> Done
# Epoch_DQN_MBRL_ODERNN_LunarLanderDiscrete -> Required


# @time Episodic_DQN_MF_Agent_LunarLanderDiscrete = @distributed (vcat) for i=1:4
#     agent(LunarLanderDiscrete(), AgentParameter(training_episodes=250, train_start=2, ϵ_min=exploreRate[i]))
# end
# @save "./output/LunarLanderDiscrete/_explore_Episodic_DQN_MF_Agent_LunarLanderDiscrete.bson" Episodic_DQN_MF_Agent_LunarLanderDiscrete 



# @time Episodic_DDPG_MBRL_NODE_LunarLanderDiscrete = @distributed (vcat) for i=1:4
#     MBRLAgent(NODEModel(), LunarLanderDiscrete(), AgentParameter(training_episodes=250, train_start=2, ϵ_min=exploreRate[i]), ModelParameter(retrain = 2000))
# end
# @save "./output/LunarLanderDiscrete/_explore_Episodic_DDPG_MBRL_NODE_LunarLanderDiscrete.bson" Episodic_DDPG_MBRL_NODE_LunarLanderDiscrete 



# @time Episodic_DQN_MBRL_ODERNN_LunarLanderDiscrete = @distributed (vcat) for i=1:4
#     MBRLAgent(ODERNNModel(), LunarLanderDiscrete(), AgentParameter(training_episodes=250, train_start=2, ϵ_min=exploreRate[i]), ModelParameter(retrain = 2000))
# end
# @save "./output/LunarLanderDiscrete/_explore_Episodic_DQN_MBRL_ODERNN_LunarLanderDiscrete.bson" Episodic_DQN_MBRL_ODERNN_LunarLanderDiscrete 


@time Episodic_DDPG_MBRL_NODE_LunarLanderDiscrete = @distributed (vcat) for i=1:10
    MBRLAgent(NODEModel(), LunarLanderDiscrete(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain = 5000))
end
@save "./output/LunarLanderDiscrete/Episodic_DDPG_MBRL_NODE_LunarLanderDiscrete.bson" Episodic_DDPG_MBRL_NODE_LunarLanderDiscrete 


