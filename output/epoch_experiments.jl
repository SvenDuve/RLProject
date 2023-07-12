using MBRL
using RLTypes
using DDPG
using DQN
using BSON: @load
using BSON

using Plots
gr()
using DataFrames
using Statistics
using StatsBase


# DQN
# Acrobot, running on remote machine
# Episode_DQN_MF_Agent_Acrobot -> Done
# Episode_DQN_MBRL_NODE_Acrobot -> Done
# Episode_DQN_MBRL_ODERNN_Acrobot -> Done


# LunarLanderDiscrete requires running on MAC
# Epoch_DQN_MF_Agent_LunarLanderDiscrete -> Done
# Epoch_DQN_MBRL_NODE_LunarLanderDiscrete -> Done
# Epoch_DQN_MBRL_ODERNN_LunarLanderDiscrete -> Done


# DDPG
# Pendulum, all running on remote machine
# Episode_DDPG_MF_Agent_Pendulum -> Done
# Episode_DDPG_MBRL_NODE_Pendulum -> Done
# Episode_DDPG_MBRL_ODERNN_Pendulum -> Done

# LunarLanderContinuous all running on remote machine
# Epoch_DDPG_MF_Agent_LunarLanderContinuous -> Required
# Epoch_DDPG_MBRL_NODE_LunarLanderContinuous -> Required
# Epoch_DDPG_MBRL_ODERNN_LunarLanderContinuous -> Required





#Epoch helper function
function sum_interval(arr::Array, interval::Int)
    sums = []
    for i in 1:interval:length(arr)
        push!(sums, sum(arr[i:min(i+interval-1, end)]))
    end
    return sums
end


# potentially rerun Model Free checking episode number


function episodicMFData(file; window_size=10)

    data = BSON.load(file)
    df = DataFrame()

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            df[!, colname] = item.episode_reward
        end
    end

    df[!, :mean] = median.(eachrow(df))
    df[!, :upper] = quantile.(eachrow(df), 0.75)
    df[!, :lower] = quantile.(eachrow(df), 0.25)
    
        
    df_ma = DataFrame()
    df_ma[!, :mean] = [mean(df.mean[i:min(i+window_size-1, end)]) for i = 1:size(df.mean, 1)-window_size+1]
    df_ma[!, :upper] = [mean(df.upper[i:min(i+window_size-1, end)]) for i = 1:size(df.upper, 1)-window_size+1]
    df_ma[!, :lower] = [mean(df.lower[i:min(i+window_size-1, end)]) for i = 1:size(df.lower, 1)-window_size+1]


    return df_ma


end


function epochMFData(file, interval; window_size=10)

    data = BSON.load(file)
    df = DataFrame()

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            #df[!, colname] = item.episode_reward
            df[!, colname] = sum_interval(item.all_rewards, interval)
        end
    end


    df[!, :mean] = median.(eachrow(df))
    df[!, :upper] = quantile.(eachrow(df), 0.75)
    df[!, :lower] = quantile.(eachrow(df), 0.25)
    
        
    df_ma = DataFrame()
    df_ma[!, :mean] = [mean(df.mean[i:min(i+window_size-1, end)]) for i = 1:size(df.mean, 1)-window_size+1]
    df_ma[!, :upper] = [mean(df.upper[i:min(i+window_size-1, end)]) for i = 1:size(df.upper, 1)-window_size+1]
    df_ma[!, :lower] = [mean(df.lower[i:min(i+window_size-1, end)]) for i = 1:size(df.lower, 1)-window_size+1]


    return df_ma


end


function episodicMBData(file; window_size=10)

    data = BSON.load(file)
    df = DataFrame()

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            df[!, colname] = item[1].episode_reward
        end
    end

    df[!, :mean] = mean.(eachrow(df))
    df[!, :upper] = quantile.(eachrow(df), 0.75)
    df[!, :lower] = quantile.(eachrow(df), 0.25)
    
        
    df_ma = DataFrame()
    df_ma[!, :mean] = [mean(df.mean[i:min(i+window_size-1, end)]) for i = 1:size(df.mean, 1)-window_size+1]
    df_ma[!, :upper] = [mean(df.upper[i:min(i+window_size-1, end)]) for i = 1:size(df.upper, 1)-window_size+1]
    df_ma[!, :lower] = [mean(df.lower[i:min(i+window_size-1, end)]) for i = 1:size(df.lower, 1)-window_size+1]


    return df_ma


end



function epochMBData(file, interval; window_size=10)

    data = BSON.load(file)
    df = DataFrame()

    for el in values(data)
        for (i, item) in enumerate(el) 
            colname = "Rewards_$i"
            #df[!, colname] = item.episode_reward
            df[!, colname] = sum_interval(item[1].all_rewards, interval)
        end
    end


    df[!, :mean] = mean.(eachrow(df))
    df[!, :upper] = quantile.(eachrow(df), 0.75)
    df[!, :lower] = quantile.(eachrow(df), 0.25)
    
        
    df_ma = DataFrame()
    df_ma[!, :mean] = [mean(df.mean[i:min(i+window_size-1, end)]) for i = 1:size(df.mean, 1)-window_size+1]
    df_ma[!, :upper] = [mean(df.upper[i:min(i+window_size-1, end)]) for i = 1:size(df.upper, 1)-window_size+1]
    df_ma[!, :lower] = [mean(df.lower[i:min(i+window_size-1, end)]) for i = 1:size(df.lower, 1)-window_size+1]


    return df_ma



end





# Acrobot

episodic_acrobot_discrete_mf = "output/Acrobot/Episode_DQN_MF_Agent_Acrobot.bson"
episodic_acrobot_discrete_odernn = "output/Acrobot/Episode_DQN_MBRL_ODERNN_Acrobot.bson"
episodic_acrobot_discrete_node = "output/Acrobot/Episode_DQN_MBRL_NODE_Acrobot.bson"



acrobot_mf = episodicMFData(episodic_acrobot_discrete_mf)
acrobot_odernn = episodicMBData(episodic_acrobot_discrete_odernn)
acrobot_node = episodicMBData(episodic_acrobot_discrete_node)


p1 = plot(acrobot_mf.mean, 
                    color = :blue, 
                    label="DQN", 
                    xlims=(0,500), 
                    xticks=0:100:500, 
                    ylims=(-600, 0), 
                    yticks=-600:100:0, 
                    legend=:bottomright,
                    legendfontsize=6,
                    title = "Acrobot",
                    titlefontsize=10, 
                    xlabel="Episode", 
                    ylabel="Rewards", 
                    guidefontsize=8)
p1 = plot!(acrobot_mf.lower, fillrange=acrobot_mf.upper, color=:blue, fillalpha=0.2, linealpha=0.0, label="IQR Range")
p1 = plot!(acrobot_odernn.mean, color = :red, label="ODE-RNN")
p1 = plot!(acrobot_odernn.lower, fillrange=acrobot_odernn.upper, color=:red, fillalpha=0.2, linealpha=0.0, label="IQR Range")
p1 = plot!(acrobot_node.mean, color = :green, label="NODE")
p1 = plot!(acrobot_node.lower, fillrange=acrobot_node.upper, color=:green, fillalpha=0.2, linealpha=0.0, label="IQR Range")

savefig("output/Acrobot/Episodic_Acrobot_Composite.png")



# Pendulum

episodic_pendulum_continuous_mf = "output/Pendulum/Episode_DDPG_MF_Agent_Pendulum.bson"
episodic_pendulum_continuous_odernn = "output/Pendulum/Episode_DDPG_MBRL_ODERNN_Pendulum.bson"
episodic_pendulum_continuous_node = "output/Pendulum/Episode_DDPG_MBRL_NODE_Pendulum.bson"

pendulum_mf = episodicMFData(episodic_pendulum_continuous_mf)
pendulum_odernn = episodicMBData(episodic_pendulum_continuous_odernn)
pendulum_node = episodicMBData(episodic_pendulum_continuous_node)


p2 = plot(pendulum_mf.mean, 
                        color = :blue, 
                        label="DDPG", 
                        xlims=(0,500), 
                        xticks=0:100:500, 
                        ylims=(-1600, 0), 
                        yticks=-1600:200:0, 
                        legend=:bottomright,
                        legendfontsize=6,
                        title = "Pendulum",
                        titlefontsize=10, 
                        xlabel="Episode", 
                        ylabel="Rewards", 
                        guidefontsize=8)
p2 = plot!(pendulum_mf.lower, fillrange=pendulum_mf.upper, color=:blue, fillalpha=0.2, linealpha=0.0, label="IQR Range")
p2 = plot!(pendulum_odernn.mean, color = :red, label="ODE-RNN")
p2 = plot!(pendulum_odernn.lower, fillrange=pendulum_odernn.upper, color=:red, fillalpha=0.2, linealpha=0.0, label="IQR Range")
p2 = plot!(pendulum_node.mean, color = :green, label="NODE")
p2 = plot!(pendulum_node.lower, fillrange=pendulum_node.upper, color=:green, fillalpha=0.2, linealpha=0.0, label="IQR Range")
# Customize the plot

savefig("output/Pendulum/Episodic_Pendulum_Composite.png")





# LunarLanderDiscrete

epoch_lunarlander_discrete_mf = "output/LunarLanderDiscrete/Epoch_DQN_MF_Agent_LunarLanderDiscrete.bson"
epoch_lunarlander_discrete_odernn = "output/LunarLanderDiscrete/Epoch_DQN_MBRL_ODERNN_LunarLanderDiscrete.bson"
epoch_lunarlander_discrete_node = "output/LunarLanderDiscrete/Epoch_DQN_MBRL_NODE_LunarLanderDiscrete.bson"



ll_discrete_mf = epochMFData(epoch_lunarlander_discrete_mf, 1000)
ll_discrete_mb_odernn = epochMBData(epoch_lunarlander_discrete_odernn, 1000)
ll_discrete_mb_node = epochMBData(epoch_lunarlander_discrete_node, 1000)


p3 = plot(ll_discrete_mf.mean, 
                        color = :blue, 
                        label="DQN", 
                        xlims=(0,200), 
                        xticks=0:50:200, 
                        ylims=(-1000, 600), 
                        yticks=-1000:200:600, 
                        legend=:bottomright,
                        legendfontsize=6,
                        title = "LunarLander",
                        titlefontsize=10, 
                        xlabel="Epoch", 
                        ylabel="Rewards", 
                        guidefontsize=8)
p3 = plot!(ll_discrete_mf.lower, fillrange=ll_discrete_mf.upper, color=:blue, fillalpha=0.2, linealpha=0.0, label="IQR Range")
p3 = plot!(ll_discrete_mb_odernn.mean, color = :red, label="ODE-RNN")
p3 = plot!(ll_discrete_mb_odernn.lower, fillrange=ll_discrete_mb_odernn.upper, color=:red, fillalpha=0.2, linealpha=0.0, label="IQR Range")
p3 = plot!(ll_discrete_mb_node.mean, color = :green, label="NODE")
p3 = plot!(ll_discrete_mb_node.lower, fillrange=ll_discrete_mb_node.upper, color=:green, fillalpha=0.2, linealpha=0.0, label="IQR Range")

savefig("output/LunarLanderDiscrete/Epoch_LunarLanderDiscrete_Composite.png")


#LunarLanderContinuous

epoch_lunarlander_continuous_mf = "output/LunarLanderContinuous/Epoch_DDPG_MF_Agent_LunarLanderContinuous.bson"
epoch_lunarlander_continuous_odernn = "output/LunarLanderContinuous/Epoch_DDPG_MBRL_ODERNN_LunarLanderContinuous.bson"
epoch_lunarlander_continuous_node = "output/LunarLanderContinuous/Epoch_DDPG_MBRL_NODE_LunarLanderContinuous.bson"


ll_continuous_mf = epochMFData(epoch_lunarlander_continuous_mf, 1000)
ll_continuous_mb_odernn = epochMBData(epoch_lunarlander_continuous_odernn, 1000)
ll_continuous_mb_node = epochMBData(epoch_lunarlander_continuous_node, 1000)


p4 = plot(ll_continuous_mf.mean, 
                        color = :blue, 
                        label="DDPG", 
                        xlims=(0,200), 
                        xticks=0:50:200, 
                        ylims=(-1600, 400), 
                        yticks=-1600:200:400, 
                        legend=:bottomright,
                        legendfontsize=6,
                        title = "LunarLander",
                        titlefontsize=10, 
                        xlabel="Epoch", 
                        ylabel="Rewards", 
                        guidefontsize=8)
p4 = plot!(ll_continuous_mf.lower, fillrange=ll_continuous_mf.upper, color=:blue, fillalpha=0.2, linealpha=0.0, label="IQR Range")
p4 = plot!(ll_continuous_mb_odernn.mean, color = :red, label="ODE-RNN")
p4 = plot!(ll_continuous_mb_odernn.lower, fillrange=ll_continuous_mb_odernn.upper, color=:red, fillalpha=0.2, linealpha=0.0, label="IQR Range")
p4 = plot!(ll_continuous_mb_node.mean, color = :green, label="NODE")
p4 = plot!(ll_continuous_mb_node.lower, fillrange=ll_continuous_mb_node.upper, color=:green, fillalpha=0.2, linealpha=0.0, label="IQR Range")


savefig("output/LunarLanderContinuous/Epoch_LunarLanderContinuous_Composite.png")



plot(p1, p3, layout=@layout([_ a{0.45w, 0.95h} _ b{0.45w} _]), size=(900, 400))
savefig("output/ComparisonDiscrete.png")

plot(p2, p4, layout=@layout([_ a{0.45w, 0.95h} _ b{0.45w} _]), size=(900, 400))
savefig("output/ComparisonContinuous.png")

















episodic_acrobot_discrete_mf = "output/Acrobot/Episode_DQN_MF_Agent_Acrobot.bson"
episodic_acrobot_discrete_odernn = "output/Acrobot/Episode_DQN_MBRL_ODERNN_Acrobot.bson"
episodic_acrobot_discrete_node = "output/Acrobot/Episode_DQN_MBRL_NODE_Acrobot.bson"



data = BSON.load(episodic_acrobot_discrete_node)
df = DataFrame()

for el in values(data)
    for (i, item) in enumerate(el) 
        colname = "Rewards_$i"
        df[!, colname] = item[1].episode_reward
    end
end


CSV.write("acrobot_node.csv", df)


episodic_pendulum_continuous_mf = "output/Pendulum/Episode_DDPG_MF_Agent_Pendulum.bson"
episodic_pendulum_continuous_odernn = "output/Pendulum/Episode_DDPG_MBRL_ODERNN_Pendulum.bson"
episodic_pendulum_continuous_node = "output/Pendulum/Episode_DDPG_MBRL_NODE_Pendulum.bson"

data = BSON.load(episodic_pendulum_continuous_node)
df = DataFrame()

for el in values(data)
    for (i, item) in enumerate(el) 
        colname = "Rewards_$i"
        df[!, colname] = item[1].episode_reward
    end
end

CSV.write("pendulum_node.csv", df)



epoch_lunarlander_discrete_mf = "output/LunarLanderDiscrete/Epoch_DQN_MF_Agent_LunarLanderDiscrete.bson"
epoch_lunarlander_discrete_odernn = "output/LunarLanderDiscrete/Epoch_DQN_MBRL_ODERNN_LunarLanderDiscrete.bson"
epoch_lunarlander_discrete_node = "output/LunarLanderDiscrete/Epoch_DQN_MBRL_NODE_LunarLanderDiscrete.bson"

data = BSON.load(epoch_lunarlander_discrete_node)
df = DataFrame()

for el in values(data)
    for (i, item) in enumerate(el) 
        colname = "Rewards_$i"
        #df[!, colname] = item.episode_reward
        df[!, colname] = sum_interval(item[1].all_rewards, 2000)
    end
end



CSV.write("lunarlanderdiscrete_node.csv", df)



epoch_lunarlander_continuous_mf = "output/LunarLanderContinuous/Epoch_DDPG_MF_Agent_LunarLanderContinuous.bson"
epoch_lunarlander_continuous_odernn = "output/LunarLanderContinuous/Epoch_DDPG_MBRL_ODERNN_LunarLanderContinuous.bson"
epoch_lunarlander_continuous_node = "output/LunarLanderContinuous/Epoch_DDPG_MBRL_NODE_LunarLanderContinuous.bson"


data = BSON.load(epoch_lunarlander_continuous_node)
df = DataFrame()

for el in values(data)
    for (i, item) in enumerate(el) 
        colname = "Rewards_$i"
        #df[!, colname] = item.episode_reward
        df[!, colname] = sum_interval(item[1].all_rewards, 2000)
    end
end



CSV.write("lunarlandercontinuous_node.csv", df)
