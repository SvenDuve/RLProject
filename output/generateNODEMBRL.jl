#julia -p 6 ./output/generateNODEMBRL.jl

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



# @time DDPG_MBRL_NODE_Pendulum = @distributed (vcat) for i=1:10
#     MBRLAgent(NODEModel(), Pendulum(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain = 5000))
# end

# @save "./output/Pendulum/DDPG_MBRL_NODE_Pendulum.bson" DDPG_MBRL_NODE_Pendulum 


@time DDPG_MBRL_NODE_LunarLanderContinuous = @distributed (vcat) for i=1:10
    MBRLAgent(NODEModel(), LunarLanderContinuous(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain = 5000))
end


@save "./output/LunarLanderContinuous/DDPG_MBRL_NODE_LunarLanderContinuous.bson" DDPG_MBRL_NODE_LunarLanderContinuous 



