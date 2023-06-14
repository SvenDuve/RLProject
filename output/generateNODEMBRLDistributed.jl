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

MBRL_NODE_Acrobot = []


@sync @distributed for i=1:10
    push!(MBRL_NODE_Acrobot, MBRLAgent(NODEModel(), Acrobot(), AgentParameter(ϵ_greedy_reduction=30000, training_episodes=500, train_start=2), ModelParameter(retrain=5000)))
end



@save "MBRL_NODE_Acrobot.bson" MBRL_NODE_Acrobot 