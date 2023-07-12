#julia -p 9 ./output/generatePendulum.jl

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
end


# Episode_DQN_MF_Agent_Acrobot -> Required
# Episode_DQN_MBRL_NODE_Acrobot -> Done
# Episode_DQN_MBRL_ODERNN_Acrobot -> Required



@time Episode_DQN_MF_Agent_Acrobot = @distributed (vcat) for i=1:10
    agent(Acrobot(), AgentParameter(training_episodes=500, train_start=2))
end
@save "./output/Acrobot/Episode_DQN_MF_Agent_Acrobot.bson" Episode_DQN_MF_Agent_Acrobot 




# @time Episode_DDPG_MBRL_NODE_Pendulum = @distributed (vcat) for i=1:10
#     MBRLAgent(NODEModel(), Pendulum(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain = 5000))
# end
# @save "./output/Pendulum/Episode_DDPG_MBRL_NODE_Pendulum.bson" Episode_DDPG_MBRL_NODE_Pendulum 




@time Episode_DQN_MBRL_ODERNN_Acrobot = @distributed (vcat) for i=1:10
    MBRLAgent(ODERNNModel(), Acrobot(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain = 5000))
end
@save "./output/Acrobot/Episode_DQN_MBRL_ODERNN_Acrobot.bson" Episode_DQN_MBRL_ODERNN_Acrobot 




