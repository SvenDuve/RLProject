using Conda
using PyCall
using RLTypes
using Rewards

gym = pyimport("gymnasium")
env = gym.make("Acrobot-v1")

s, info = env.reset()

for i in 1:10
    a = env.action_space.sample()
    s_next, r, t, _ = env.step(a)
    @show rew(Acrobot(), s, a, s_next)
    @show r
    println(s)
    s = s_next
end


n = 10^6
1000000


d = @distributed (+) for i=1:10
           using DQN, RLTypes, Rewards
           agent(Acrobot(), AgentParameter())
    end
333333833333500000

julia> println(d)
333333833333500000

@sync @distributed for i=1:3
    agent(Acrobot(), AgentParameter())
end




@distributed for i=1:3
    agent(Acrobot(), AgentParameter())
end