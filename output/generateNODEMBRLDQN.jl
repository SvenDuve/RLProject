#julia -p 6 ./output/generateNODEMBRLDQN.jl

using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin 
    using RLTypes
    using MBRL
    using BSON: @save 
end



# @time Episode_DQN_MBRL_NODE_Acrobot = @distributed (vcat) for i=1:10
#     MBRLAgent(NODEModel(), Acrobot(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain = 5000))
# end

# @save "./output/Acrobot/Episode_DQN_MBRL_NODE_Acrobot.bson" Episode_DQN_MBRL_NODE_Acrobot 


@time Epoch_DQN_MBRL_NODE_LunarLanderDiscrete = @distributed (vcat) for i=1:2
    MBRLAgent(NODEModel(), LunarLanderDiscrete(), AgentParameter(train_type = Epoch(), training_epochs=10, train_start=2), ModelParameter(retrain = 5000))
end


@save "./output/LunarLanderDiscrete/_Epoch_DQN_MBRL_NODE_LunarLanderDiscrete.bson" Epoch_DQN_MBRL_NODE_LunarLanderDiscrete 



