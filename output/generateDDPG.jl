#julia -p 6 ./output/_generateNODEMBRLDistrbuted.jl

using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin 
    using RLTypes
    using DDPG
    using BSON: @save 
end


@time DDPG_MF_Agent_Pendulum = @distributed (vcat) for i=1:10
    agent(Pendulum(), AgentParameter(training_episodes=500, train_start=2))
end

@save "./output/Pendulum/DDPG_MF_Agent_Pendulum.bson" DDPG_MF_Agent_Pendulum 




@time DDPG_MF_Agent_LunarLanderContinuous = @distributed (vcat) for i=1:10
    agent(LunarLanderContinuous(), AgentParameter(training_episodes=500, train_start=2))
end

@save "./output/LunarLanderContinuous/DDPG_MF_Agent_LunarLanderContinuous.bson" DDPG_MF_Agent_LunarLanderContinuous 

