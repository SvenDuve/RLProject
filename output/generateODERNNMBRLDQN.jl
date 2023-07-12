#julia -p 4 ./output/generateODERNNMBRLDQN.jl

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



@time DQN_MBRL_ODERNN_Acrobot = @distributed (vcat) for i=1:10
    MBRLAgent(ODERNNModel(), Acrobot(), AgentParameter(train_type=Episode(), training_episodes=500, train_start=2), ModelParameter(retrain = 5000))
end

@save "./output/Acrobot/Episode_DQN_MBRL_ODERNN_Acrobot.bson" DQN_MBRL_ODERNN_Acrobot 


@time DQN_MBRL_ODERNN_LunarLanderDiscrete = @distributed (vcat) for i=1:10
    MBRLAgent(ODERNNModel(), LunarLanderDiscrete(), AgentParameter(train_type=Epoch(), training_episodes=500, train_start=2), ModelParameter(retrain = 5000))
end


@save "./output/LunarLanderDiscrete/DQN_MBRL_ODERNN_LunarLanderDiscrete.bson" DQN_MBRL_ODERNN_LunarLanderDiscrete 



