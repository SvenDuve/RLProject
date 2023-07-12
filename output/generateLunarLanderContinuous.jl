

using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin 
    using RLTypes
    using DDPG
    using MBRL
    using BSON: @save 
end


# Epoch_DDPG_MF_Agent_LunarLanderContinuous -> Required
# Epoch_DDPG_MBRL_NODE_LunarLanderContinuous -> Required
# Epoch_DDPG_MBRL_ODERNN_LunarLanderContinuous -> Required



@time Episodic_DDPG_MF_Agent_LunarLanderContinuous = @distributed (vcat) for i=1:10
    agent(LunarLanderContinuous(), AgentParameter(training_episodes = 500, train_start=2))
end
@save "./output/LunarLanderContinuous/Episodic_DDPG_MF_Agent_LunarLanderContinuous.bson" Episodic_DDPG_MF_Agent_LunarLanderContinuous 




@time Episodic_DDPG_MBRL_NODE_LunarLanderContinuous = @distributed (vcat) for i=1:10
    MBRLAgent(NODEModel(), LunarLanderContinuous(), AgentParameter(training_episodes= 500, train_start=2), ModelParameter(retrain = 5000))
end
@save "./output/LunarLanderContinuous/Episodic_DDPG_MBRL_NODE_LunarLanderContinuous.bson" Episodic_DDPG_MBRL_NODE_LunarLanderContinuous 




@time Episodic_DDPG_MBRL_ODERNN_LunarLanderContinuous = @distributed (vcat) for i=1:10
    MBRLAgent(ODERNNModel(), LunarLanderContinuous(), AgentParameter(training_episodes= 500, train_start=2), ModelParameter(retrain = 5000))
end
@save "./output/LunarLanderContinuous/Episodic_DDPG_MBRL_ODERNN_LunarLanderContinuous.bson" Episodic_DDPG_MBRL_ODERNN_LunarLanderContinuous 




