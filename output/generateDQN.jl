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

@time DQN_MF_Agent_Acrobot = @distributed (vcat) for i=1:10
    agent(Acrobot(), AgentParameter(training_episodes=500, train_start=2))
end


@save "DQN_MF_Agent_Acrobot.bson" DQN_MF_Agent_Acrobot 


@time DQN_MF_Agent_LunarLanderDiscrete = @distributed (vcat) for i=1:10
    agent(LunarLanderDiscrete(), AgentParameter(training_episodes=500, train_start=2))
end


@save "DQN_MF_Agent_LunarLanderDiscrete.bson" DQN_MF_Agent_LunarLanderDiscrete 