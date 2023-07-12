using MBRL
using RLTypes
using DDPG
using DQN
using BSON: @load
using BSON

using Plots
using DataFrames
using Statistics


# Databases created:
# LunarLanderContinuous:
# DDPG_MF_Agent_LunarLanderContinuous, This is the plain Model Free DDPG Agent, Function Call: agent(LunarLanderContinuous(), AgentParameter(training_episodes=500, train_start=2))
# DDPG_MBRL_NODE_LunarLanderContinuous, running, function call: MBRLAgent(NODEModel(), LunarLanderContinuous(), AgentParameter(ϵ_greedy_reduction=30000, training_episodes=500, train_start=2), ModelParameter(retrain=5000))
# DDPG_MBRL_ODERNN_LunarLanderContinuous, done on the Macbook, function call: 




# Pendulum
# DDPG_MD_Agent_Pendulum, this is the model free pendulum agent, DDPG, Function Call: agent(Pendulum(), AgentParameter(training_episodes=500, train_start=2))
# DDPG_MBRL_NODE_Pendulum, running, function call: MBRLAgent(NODEModel(), Pendulum(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain = 5000))
# DDPG_MBRL_ODERNN_Pendulum, running





# LunarLanderDiscrete
# DQN_MF_Agent_LunarLanderDiscrete, model free learner acrobot, DQN, function call: agent(LunarLanderDiscrete(), AgentParameter(training_episodes=500, train_start=2))
# DQN_MBRL_NODE_LunarLanderDiscrete, model based, Dynode type, funciton call: ___running___ MBRLAgent(NODEModel(), LunarLanderDiscrete(), AgentParameter(ϵ_greedy_reduction=30000, training_episodes=500, train_start=2), ModelParameter(retrain=5000))
# DQN_MBRL_ODERNN_LunarLanderDiscrete, function call: MBRLAgent(ODERNNModel(), LunarLanderDiscrete(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain = 5000))





# Acrobot
# DQN_MF_Agent_Acrobot, moder free learner function call: agent(Acrobot(), AgentParameter(training_episodes=500, train_start=2))
# DQN_MBRL_NODE_Acrobot, model based dynde type, function call: MBRLAgent(NODEModel(), Acrobot(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain=5000))
# DQN_MBRL_ODERNN_Acrobot, model based ODERNN style, function call: MBRLAgent(ODERNNModel(), Acrobot(), AgentParameter(training_episodes=500, train_start=2), ModelParameter(retrain = 5000))



struct ModelFree end
struct ModelBased end


function plot_agents(agent::ModelBased, file, title, label; ylabel="Rewards", xlabel="Episodes")
    
    data = BSON.load(file)
    df = DataFrame()

    # Create column names and data in a for loop

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            df[!, colname] = item[1].episode_reward
        end
    end

    df[!, :mean] = mean.(eachrow(df))
    df[!, :max] = maximum.(eachrow(df))
    df[!, :min] = minimum.(eachrow(df))

    pl = plot(df.mean, color = :blue, title=title, label=label, ylabel=ylabel, xlabel=xlabel)
    pl = plot!(df.min, fillrange=df.max, color=:blue, fillalpha=0.2, linealpha=0.0, label="Max - Min")
    # savefig("output/Acrobot/DQN_Acrobot.png")

    return pl
end

function plot_mean_agents(agent::ModelBased, file, title, label; ylabel="Rewards", xlabel="Episodes")
    
    data = BSON.load(file)
    df = DataFrame()

    # Create column names and data in a for loop

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            df[!, colname] = item[1].episode_reward
        end
    end

    df[!, :mean] = mean.(eachrow(df))
    df[!, :max] = maximum.(eachrow(df))
    df[!, :min] = minimum.(eachrow(df))

    pl = plot(df.mean, color = :blue, title=title, label=label, ylabel=ylabel, xlabel=xlabel)
    # savefig("output/Acrobot/DQN_Acrobot.png")

    return pl
end



function plot_agents(agent::ModelFree, file, title, label; ylabel="Rewards", xlabel="Episodes")
    
    data = BSON.load(file)
    df = DataFrame()

    # Create column names and data in a for loop

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            df[!, colname] = item.episode_reward
        end
    end

    df[!, :mean] = mean.(eachrow(df))
    df[!, :max] = maximum.(eachrow(df))
    df[!, :min] = minimum.(eachrow(df))

    pl = plot(df.mean, color = :blue, title=title, label=label, ylabel=ylabel, xlabel=xlabel)
    pl = plot!(df.min, fillrange=df.max, color=:blue, fillalpha=0.2, linealpha=0.0, label="Max - Min")
    # savefig("output/Acrobot/DQN_Acrobot.png")

    return pl
end

function plot_mean_agents(agent::ModelFree, file, title, label; ylabel="Rewards", xlabel="Episodes")
    
    data = BSON.load(file)
    df = DataFrame()

    # Create column names and data in a for loop

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            df[!, colname] = item.episode_reward
        end
    end

    df[!, :mean] = mean.(eachrow(df))
    df[!, :max] = maximum.(eachrow(df))
    df[!, :min] = minimum.(eachrow(df))

    pl = plot(df.mean, color = :blue, title=title, label=label, ylabel=ylabel, xlabel=xlabel)
    # savefig("output/Acrobot/DQN_Acrobot.png")
    return pl
end


function plot_comparison(baseline, modelbased, title, series_one, series_two; ylabel="Rewards", xlabel="Episodes")
    baseline_data = BSON.load(baseline)
    baseline_df = DataFrame()

    modelbased_data = BSON.load(modelbased)
    modelbased_df = DataFrame()

    for el in values(baseline_data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            baseline_df[!, colname] = item.episode_reward
        end
    end

    baseline_df[!, :mean] = mean.(eachrow(baseline_df))

    for el in values(modelbased_data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            modelbased_df[!, colname] = item[1].episode_reward
        end
    end

    modelbased_df[!, :mean] = mean.(eachrow(modelbased_df))

    pl = plot(baseline_df.mean, color = :blue, title=title, label=series_one, ylabel=ylabel, xlabel=xlabel)  
    pl = plot!(modelbased_df.mean, color=:green, label=series_two)

    return pl


end

file = "output/LunarLanderDiscrete/Epoch_DQN_MBRL_NODE_LunarLanderDiscrete.bson"
data = BSON.load(file)
plot(sum_interval(data[:Epoch_DQN_MBRL_NODE_LunarLanderDiscrete][1][1].all_rewards, 1000))



file_2 = "Epoch_DQN_MF_Agent_LunarLanderDiscrete.bson"
data_2 = BSON.load(file_2)
plot(sum_interval(data_2[:Epoch_DQN_MF_Agent_LunarLanderDiscrete][1].all_rewards, 1000))


#Epoch helper function
function sum_interval(arr::Array, interval::Int)
    sums = []
    for i in 1:interval:length(arr)
        push!(sums, sum(arr[i:min(i+interval-1, end)]))
    end
    return sums
end


function epochChartMB(file, interval, epochs)
    
    data = BSON.load(file)
    df = DataFrame()

    # Create column names and data in a for loop

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            df[!, colname] = sum_interval(item[1].all_rewards[1:150000], interval)
        end
    end

    @show df

    df[!, :mean] = mean.(eachrow(df))
    df[!, :max] = maximum.(eachrow(df))
    df[!, :min] = minimum.(eachrow(df))

    pl = plot(df.mean, color = :blue, title="Lunar Lander DDPG Reward", label="DDPG")
    pl = plot!(df.min, fillrange=df.max, color=:blue, fillalpha=0.2, linealpha=0.0, label="Max - Min")
    # savefig("output/Acrobot/DQN_Acrobot.png")

    return pl
end


function epochChartMF(file, interval, epochs)
    
    data = BSON.load(file)
    df = DataFrame()

    # Create column names and data in a for loop

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            df[!, colname] = sum_interval(item.all_rewards[1:150000], interval)
        end
    end

    df[!, :mean] = mean.(eachrow(df))
    df[!, :max] = maximum.(eachrow(df))
    df[!, :min] = minimum.(eachrow(df))

    pl = plot(df.mean, color = :blue, title="Lunar Lander DDPG Reward", label="DDPG")
    pl = plot!(df.min, fillrange=df.max, color=:blue, fillalpha=0.2, linealpha=0.0, label="Max - Min")
    # savefig("output/Acrobot/DQN_Acrobot.png")

    return pl
end


epochChartMB(file, 3000, 1000)
epochChartMF(file_2, 3000, 1000)



# Acrobot Reward Curves

acrobot_mf = "output/Acrobot/DQN_MF_Agent_Acrobot.bson"
acrobot_odernn = "output/Acrobot/DQN_MBRL_ODERNN_Acrobot.bson"
acrobot_node = "output/Acrobot/DQN_MBRL_NODE_Acrobot.bson"
epoch_acrobot_node = "output/Acrobot/Epoch_DQN_MBRL_NODE_Acrobot.bson"



p1 = plot_agents(ModelFree(), acrobot_mf, "Acrobot", "DQN")
p2 = plot_agents(ModelBased(), acrobot_odernn, "Acrobot", "ODE-RNN")
p3 = plot_agents(ModelBased(), acrobot_node, "Acrobot", "NODE")

p4 = plot_comparison(acrobot_mf, acrobot_odernn, "Acrobot", "DQN", "ODE-RNN")
p5 = plot_comparison(acrobot_mf, acrobot_node, "Acrobot", "DQN", "NODE")

l = @layout [
    [grid(1,3)]
    [grid(1,2)]
]

plot(p1, p2, p3, p4, p5, layout=l, size=(1600, 1200))

savefig("output/Acrobot/Acrobot_Comparison.png")



# Lunar Lander Discrete

lunarlander_discrete_mf = "output/LunarLanderDiscrete/DQN_MF_Agent_LunarLanderDiscrete.bson"
lunarlander_discrete_odernn = "./output/LunarLanderDiscrete/DQN_MBRL_ODERNN_LunarLanderDiscrete.bson"
lunarlander_discrete_node = "output/LunarLanderDiscrete/MBRL_NODE_LunarLanderDiscrete.bson"


p1 = plot_agents(ModelFree(), lunarlander_discrete_mf, "Lunar Lander Discrete DQN", "DQN")
p2 = plot_agents(ModelBased(), lunarlander_discrete_odernn, "Lunar Lander Discrete ODE-RNN", "ODE-RNN")
p3 = plot_agents(ModelBased(), lunarlander_discrete_node, "Lunar Lander Discrete NODE", "NODE")

p4 = plot_comparison(lunarlander_discrete_mf, lunarlander_discrete_odernn, "Lunar Lander Discrete", "DQN", "ODE-RNN")
p5 = plot_comparison(lunarlander_discrete_mf, lunarlander_discrete_node, "Lunar Lander Discrete", "DQN", "NODE")

l = @layout [
    [grid(1,3)]
    [grid(1,2)]
]

plot(p1, p2, p3, p4, p5, layout=l, size=(1600, 1200))

savefig("output/LunarLanderDiscrete/LunarLanderDiscrete_Comparison.png")


# Pendulum Reward Curves

pendulum_mf = "output/Pendulum/DDPG_MF_Agent_Pendulum.bson"
pendulum_odernn = "output/Pendulum/DDPG_MBRL_ODERNN_Pendulum.bson"
pendulum_node = "output/Pendulum/DDPG_MBRL_NODE_Pendulum.bson"


p1 = plot_agents(ModelFree(), pendulum_mf, "DDPG Pendulum", "Average Rewards")
p2 = plot_agents(ModelBased(), pendulum_odernn, "Pendulum ODE-RNN", "Average Rewards")
p3 = plot_agents(ModelBased(), pendulum_node, "Pendulum NODE", "Average Rewards")

p4 = plot_comparison(pendulum_mf, pendulum_odernn, "Pendululm", "DDPG", "ODE_RNN")
p5 = plot_comparison(pendulum_mf, pendulum_node, "Pendululm", "DDPG", "NODE")

l = @layout [
    [grid(1,3)]
    [grid(1,2)]
]

plot(p1, p2, p3, p4, p5, layout=l, size=(1600, 1200))

savefig("output/Pendulum/Pendulum_Comparison.png")



# Lunar Lander Continuous

lunarlander_continuous_mf = "output/LunarLanderContinuous/DDPG_MF_Agent_LunarLanderContinuous.bson"
lunarlander_continuous_odernn = "output/LunarLanderContinuous/DDPG_MBRL_ODERNN_LunarLanderContinuous.bson"
lunarlander_continuous_node = "output/LunarLanderContinuous/DDPG_MBRL_NODE_LunarLanderContinuous.bson"

p1 = plot_agents(ModelFree(), lunarlander_continuous_mf, "DDPG Lunar Lander", "Average Rewards")
p2 = plot_agents(ModelBased(), lunarlander_continuous_odernn, "DDPG Lunar Lander Cont.", "ODE-RNN")
p3 = plot_agents(ModelBased(), lunarlander_continuous_node, "DDPG Lunar Lander Cont.", "NODE")

p4 = plot_comparison(lunarlander_continuous_mf, lunarlander_continuous_odernn, "DDPG vs. ODE-RNN", "DDPG", "ODE-RNN")
p5 = plot_comparison(lunarlander_continuous_mf, lunarlander_continuous_node, "DDPG vs. NODE", "DDPG", "NODE")




l = @layout [
    [grid(1,3)]
    [grid(1,2)]
]

plot(p1, p2, p3, p4, p5, layout=l, size=(1600, 1200))

savefig("output/LunarLanderContinuous/LunarLanderContinuous_Comparison.png")
























p1 = plot(ll_df.mean, color = :blue, title="Lunar Lander DDPG Reward", label="DDPG", ylabel="Reward", xlabel="Episodes")
p1 = plot!(ll_df.min, fillrange=ll_df.max, color=:blue, fillalpha=0.2, linealpha=0.0, label="Max - Min")

p2 = plot(mbrl_lunarlander_df.mean, color = :green, title="Lunar Lander MBRL ODE-RNN Reward", label="MBRL DDPG")
p2 = plot!(mbrl_lunarlander_df.min, fillrange=ll_df.max, color=:green, fillalpha=0.2, linealpha=0.0, label="Max - Min")


p3 = plot(mbrl_node_lunarlander_df.mean, color = :red, title="Lunar Lander MBRL NODE Reward", label="MBRL DDPG")
p3 = plot!(mbrl_node_lunarlander_df.min, fillrange=ll_df.max, color=:red, fillalpha=0.2, linealpha=0.0, label="Max - Min")

#Comparison


p4 = plot(ll_df.mean, color = :blue, lw=0.5, title="Lunar Lander MF vs. MB", label="DDPG")
p4 = plot!(mbrl_lunarlander_df.mean, color=:green, lw=0.5, label="MBRL ODE-RNN")

p5 = plot(ll_df.mean, color = :blue, lw=0.5, title="Lunar Lander MF vs. MB", label="DDPG")
p5 = plot!(mbrl_node_lunarlander_df.mean, lw=0.5, color=:red, label="MBRL NODE")


l = @layout [
    [grid(1,3)]
    [grid(1,2)]
]

plot(p1, p2, p3, p4, p5, layout=l, size=(1600, 1200))

savefig("output/LunarLander/LunarLander_Comparison.png")



# Pendulum DDPG
@load "./output/Pendulum/DDPG_MF_Agent_Pendulum.bson" DDPG_MF_Agent_Pendulum
# Create an empty DataFrame
pn_df = DataFrame()


# Create column names and data in a for loo
for i in 1:length(DDPG_MF_Agent_Pendulum)
    colname = "Rewards_$i"
    data = DDPG_MF_Agent_Pendulum[i].episode_reward
    pn_df[!, colname] = data
end

# apply the mean over each row
pn_df[!, :mean] = mean.(eachrow(pn_df))
pn_df[!, :max] = maximum.(eachrow(pn_df))
pn_df[!, :min] = minimum.(eachrow(pn_df))


p1 = plot(pn_df.mean, color = :blue, title="Pendulum DDPG Reward", label="DDPG")
p1 = plot!(pn_df.min, fillrange=pn_df.max, color=:blue, fillalpha=0.2, linealpha=0.0, label="Max - Min")
#plot!(ll_df.max, color=:blue)
savefig("output/LunarLander/DDPG_Pendulum.png")


# Lunar Lander MBRL
@load "./output/Pendulum/MBRL_Pendulum.bson" MBRL_Pendulum

mbrl_pendulum_df = DataFrame()

# Create column names and data in a for loop
for i in 1:length(MBRLAgents)
    colname = "Rewards_$i"
    data = MBRL_Pendulum[i][1].episode_reward
    mbrl_pendulum_df[!, colname] = data
end


mbrl_pendulum_df[!, :mean] = mean.(eachrow(mbrl_pendulum_df))
mbrl_pendulum_df[!, :max] = maximum.(eachrow(mbrl_pendulum_df))
mbrl_pendulum_df[!, :min] = minimum.(eachrow(mbrl_pendulum_df))


p2 = plot(mbrl_pendulum_df.mean, color = :green, title="Pendulum MBRL ODE-RNN Reward", label="MBRL DDPG")
p2 = plot!(mbrl_pendulum_df.min, fillrange=mbrl_pendulum_df.max, color=:green, fillalpha=0.2, linealpha=0.0, label="Max - Min")

savefig("output/Pendulum/MBRL_Pendulum_ODERNN.png")





# Pendulum NODE
@load "./output/Pendulum/MBRL_NODE_Pendulum.bson" MBRL_NODE_Pendulum

mbrl_node_pendulum_df = DataFrame()

# Create column names and data in a for loop
for i in 1:length(MBRLAgents)
    colname = "Rewards_$i"
    data = MBRL_NODE_Pendulum[i][1].episode_reward
    mbrl_node_pendulum_df[!, colname] = data
end


mbrl_node_pendulum_df[!, :mean] = mean.(eachrow(mbrl_node_pendulum_df))
mbrl_node_pendulum_df[!, :max] = maximum.(eachrow(mbrl_node_pendulum_df))
mbrl_node_pendulum_df[!, :min] = minimum.(eachrow(mbrl_node_pendulum_df))


p3 = plot(mbrl_node_pendulum_df.mean, color = :red, title="Pendulum MBRL NODE Reward", label="MBRL DDPG")
p3 = plot!(mbrl_node_pendulum_df.min, fillrange=mbrl_node_pendulum_df.max, color=:red, fillalpha=0.2, linealpha=0.0, label="Max - Min")

savefig("output/Pendulum/MBRL_Pendulum_ODERNN.png")




p4 = plot(pn_df.mean, color = :blue, lw=0.5, title="Model Free vs. MBRL ODE-RNN", label="DDPG")
p4 = plot!(mbrl_pendulum_df.mean, color=:green, lw=0.5, label="MBRL ODE-RNN")


p5 = plot(pn_df.mean, color = :blue, lw=0.5, title="Model Free vs. MBRL NODE", label="DDPG")
p5 = plot!(mbrl_node_pendulum_df.mean, color=:red, lw=0.5, label="MBRL NODE")



l = @layout [
    [grid(1,3)]
    [grid(1,2)]
]

plot(p1, p2, p3, p4, p5, layout=l, size=(1600, 1200))

savefig("output/Pendulum/Pendulum_Comparison.png")







# Plot DDPG Lunar Lander Discrete

@load "./output/LunarLanderDiscrete/DQN_MF_Agent_LunarLanderDiscrete.bson" DQN_MF_Agent_LunarLanderDiscrete
# Create an empty DataFrame
ll_df = DataFrame()


# Create column names and data in a for loo
for i in 1:length(DQN_MF_Agent_LunarLanderDiscrete)
    colname = "Rewards_$i"
    data = DQN_MF_Agent_LunarLanderDiscrete[i].episode_reward
    ll_df[!, colname] = data
end

# apply the mean over each row
ll_df[!, :mean] = mean.(eachrow(ll_df))
ll_df[!, :max] = maximum.(eachrow(ll_df))
ll_df[!, :min] = minimum.(eachrow(ll_df))



p1 = plot(ll_df.mean, color = :blue, title="Lunar Lander DDPG Reward", label="DDPG")
p1 = plot!(ll_df.min, fillrange=ll_df.max, color=:blue, fillalpha=0.2, linealpha=0.0, label="Max - Min")


#plot!(ll_df.max, color=:blue)
savefig("output/LunarLander/DDPG_LunarLander.png")



# Lunar Lander MBRL
@load "./output/LunarLanderDiscrete/MBRL_NODE_LunarLanderDiscrete.bson" MBRL_NODE_LunarLanderDiscrete

mbrl_lunarlander_df = DataFrame()

# Create column names and data in a for loop
for i in 1:length(MBRL_NODE_LunarLanderDiscrete)
    colname = "Rewards_$i"
    data = MBRL_NODE_LunarLanderDiscrete[i][1].episode_reward
    mbrl_lunarlander_df[!, colname] = data
end


mbrl_lunarlander_df[!, :mean] = mean.(eachrow(mbrl_lunarlander_df))
mbrl_lunarlander_df[!, :max] = maximum.(eachrow(mbrl_lunarlander_df))
mbrl_lunarlander_df[!, :min] = minimum.(eachrow(mbrl_lunarlander_df))

p2 = plot(mbrl_lunarlander_df.mean, color = :green, title="Lunar Lander MBRL ODE-RNN Reward", label="MBRL DDPG")
p2 = plot!(mbrl_lunarlander_df.min, fillrange=ll_df.max, color=:green, fillalpha=0.2, linealpha=0.0, label="Max - Min")

savefig("output/LunarLander/MBRL_LunarLander_ODERNN.png")



# Lunar Lander MBRL
@load "./output/LunarLanderContinuous/MBRL_NODE_LunarLander.bson" MBRL_NODE_LunarLander

mbrl_node_lunarlander_df = DataFrame()

# Create column names and data in a for loop
for i in 1:length(MBRL_NODE_LunarLander)
    colname = "Rewards_$i"
    data = MBRL_NODE_LunarLander[i][1].episode_reward
    mbrl_node_lunarlander_df[!, colname] = data
end


mbrl_node_lunarlander_df[!, :mean] = mean.(eachrow(mbrl_node_lunarlander_df))
mbrl_node_lunarlander_df[!, :max] = maximum.(eachrow(mbrl_node_lunarlander_df))
mbrl_node_lunarlander_df[!, :min] = minimum.(eachrow(mbrl_node_lunarlander_df))


p3 = plot(mbrl_node_lunarlander_df.mean, color = :red, title="Lunar Lander MBRL NODE Reward", label="MBRL DDPG")
p3 = plot!(mbrl_node_lunarlander_df.min, fillrange=ll_df.max, color=:red, fillalpha=0.2, linealpha=0.0, label="Max - Min")

savefig("output/LunarLander/MBRL_LunarLander_NODE.png")



#Comparison


p4 = plot(ll_df.mean, color = :blue, lw=0.5, title="Lunar Lander MF vs. MB", label="DDPG")
p4 = plot!(mbrl_lunarlander_df.mean, color=:green, lw=0.5, label="MBRL ODE-RNN")

p5 = plot(ll_df.mean, color = :blue, lw=0.5, title="Lunar Lander MF vs. MB", label="DDPG")
p5 = plot!(mbrl_node_lunarlander_df.mean, lw=0.5, color=:red, label="MBRL NODE")


l = @layout [
    [grid(1,3)]
    [grid(1,2)]
]

plot(p1, p2, p3, p4, p5, layout=l, size=(1600, 1200))

savefig("output/LunarLander/LunarLander_Comparison.png")








# Plot Acrobot

@load "./output/Acrobot/DQN_MF_Agent_Acrobot.bson" DQN_MF_Agent_Acrobot
# Create an empty DataFrame
ll_df = DataFrame()


# Create column names and data in a for loo
for i in 1:length(DQN_MF_Agent_Acrobot)
    colname = "Rewards_$i"
    data = DQN_MF_Agent_Acrobot[i].episode_reward
    ll_df[!, colname] = data
end

# apply the mean over each row
ll_df[!, :mean] = mean.(eachrow(ll_df))
ll_df[!, :max] = maximum.(eachrow(ll_df))
ll_df[!, :min] = minimum.(eachrow(ll_df))



p1 = plot(ll_df.mean, color = :blue, title="Lunar Lander DDPG Reward", label="DDPG")
p1 = plot!(ll_df.min, fillrange=ll_df.max, color=:blue, fillalpha=0.2, linealpha=0.0, label="Max - Min")


#plot!(ll_df.max, color=:blue)
savefig("output/LunarLander/DDPG_LunarLander.png")



# Lunar Lander MBRL
@load "./output/Acrobot/MBRL_NODE_Acrobot.bson" MBRL_NODE_Acrobot

mbrl_acrobot_df = DataFrame()

# Create column names and data in a for loop
for i in 1:length(MBRL_NODE_Acrobot)
    colname = "Rewards_$i"
    data = MBRL_NODE_Acrobot[i][1].episode_reward
    mbrl_acrobot_df[!, colname] = data
end


mbrl_acrobot_df[!, :mean] = mean.(eachrow(mbrl_acrobot_df))
mbrl_acrobot_df[!, :max] = maximum.(eachrow(mbrl_acrobot_df))
mbrl_acrobot_df[!, :min] = minimum.(eachrow(mbrl_acrobot_df))

p2 = plot(mbrl_acrobot_df.mean, color = :green, title="Lunar Lander MBRL ODE-RNN Reward", label="MBRL DDPG")
p2 = plot!(mbrl_acrobot_df.min, fillrange=ll_df.max, color=:green, fillalpha=0.2, linealpha=0.0, label="Max - Min")

savefig("output/LunarLander/MBRL_LunarLander_ODERNN.png")



# Lunar Lander MBRL
@load "./output/LunarLanderContinuous/MBRL_NODE_LunarLander.bson" MBRL_NODE_LunarLander

mbrl_node_lunarlander_df = DataFrame()

# Create column names and data in a for loop
for i in 1:length(MBRL_NODE_LunarLander)
    colname = "Rewards_$i"
    data = MBRL_NODE_LunarLander[i][1].episode_reward
    mbrl_node_lunarlander_df[!, colname] = data
end


mbrl_node_lunarlander_df[!, :mean] = mean.(eachrow(mbrl_node_lunarlander_df))
mbrl_node_lunarlander_df[!, :max] = maximum.(eachrow(mbrl_node_lunarlander_df))
mbrl_node_lunarlander_df[!, :min] = minimum.(eachrow(mbrl_node_lunarlander_df))


p3 = plot(mbrl_node_lunarlander_df.mean, color = :red, title="Lunar Lander MBRL NODE Reward", label="MBRL DDPG")
p3 = plot!(mbrl_node_lunarlander_df.min, fillrange=ll_df.max, color=:red, fillalpha=0.2, linealpha=0.0, label="Max - Min")

savefig("output/LunarLander/MBRL_LunarLander_NODE.png")



#Comparison


p4 = plot(ll_df.mean, color = :blue, lw=0.5, title="Lunar Lander MF vs. MB", label="DDPG")
p4 = plot!(mbrl_lunarlander_df.mean, color=:green, lw=0.5, label="MBRL ODE-RNN")

p5 = plot(ll_df.mean, color = :blue, lw=0.5, title="Lunar Lander MF vs. MB", label="DDPG")
p5 = plot!(mbrl_node_lunarlander_df.mean, lw=0.5, color=:red, label="MBRL NODE")


l = @layout [
    [grid(1,3)]
    [grid(1,2)]
]

plot(p1, p2, p3, p4, p5, layout=l, size=(1600, 1200))

savefig("output/LunarLander/LunarLander_Comparison.png")





using Rewards
using PyCall
using RLTypes

rewWalker = Reward(BipedalWalker())
rewPend = Reward(Pendulum())

gym = pyimport("gymnasium")
env = gym.make("Pendulum-v1")
s, info = env.reset()

rewPend.prev_shaping = shaping(Pendulum(), s)
rewWalker.prev_shaping = shaping(BipedalWalker(), s)

while true
    a = env.action_space.sample()
    s´, r, terminated, truncated, _ = env.step(a)
    @show r
    @show (rewPend(Pendulum(), s, a))
    if terminated | truncated
        break
    end
    s = s´
end



# a loop with two coditional break conditions

epochs = true
episodes = false
n_epochs = 0
n_episodes = 0


while true 
    println("Epochs... $n_epochs")
    println("Episodes... $n_episodes")
    if epochs
        n_epochs > 10 ? break : n_epochs += 1
    elseif episodes
        n_episodes > 10 ? break : n_episodes += 1
    end
end





