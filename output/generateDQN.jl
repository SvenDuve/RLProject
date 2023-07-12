#julia -p 9 ./output/generateDQN.jl

using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin 
    using RLTypes
    using DQN
    using BSON: @save 
end

@time Episode_DQN_MF_Agent_Acrobot = @distributed (vcat) for i=1:10
    agent(Acrobot(), AgentParameter(training_episodes=500, train_start=2))
end


@save "./output/Acrobot/Episode_DQN_MF_Agent_Acrobot.bson" Episode_DQN_MF_Agent_Acrobot 


@time Epoch_DQN_MF_Agent_LunarLanderDiscrete = @distributed (vcat) for i=1:10
    agent(LunarLanderDiscrete(), AgentParameter(train_type=Epoch(), training_epochs=100, train_start=2))
end


@save "./output/LunarLanderDiscrete/Epoch_DQN_MF_Agent_LunarLanderDiscrete.bson" Epoch_DQN_MF_Agent_LunarLanderDiscrete 