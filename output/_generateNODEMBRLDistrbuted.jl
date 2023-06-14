#julia -p 6 ./output/_generateNODEMBRLDistrbuted.jl

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

@time MBRL_NODE_LunarLanderDiscrete = @distributed (vcat) for i=1:10
    MBRLAgent(NODEModel(), LunarLanderDiscrete(), AgentParameter(ϵ_greedy_reduction=30000, training_episodes=500, train_start=2), ModelParameter(retrain=5000))
end


@save "MBRL_NODE_LunarLanderDiscrete.bson" MBRL_NODE_LunarLanderDiscrete 


@time MBRL_NODE_Acrobot = @distributed (vcat) for i=1:10
    MBRLAgent(NODEModel(), Acrobot(), AgentParameter(ϵ_greedy_reduction=30000, training_episodes=500, train_start=2), ModelParameter(retrain=5000))
end


@save "MBRL_NODE_Acrobot.bson" MBRL_NODE_Acrobot 