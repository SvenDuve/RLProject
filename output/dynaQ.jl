


using Plots
using StatsBase
using Colors

function eGreedy(q::Dict, ϵ::Float64)
	
	if rand() < ϵ
		return keys(q) |> collect |> sample
	else
		return collect(keys(q))[values(q) |> collect |> argmax]
	end

end
	
	
mutable struct MazeGrid
	# Grösse
	# Mappings
	# Rewards
	# States
	# All States
	# Actions
	l
	size
	mapping
	A::Array
	S⁺::Vector{Int64}
	
	function MazeGrid(l)
		size = l*l
		A = ["→", "↓", "↑", "←"]
		S⁺ = Vector(1:size)
		mapping = Dict([(s, Dict([(A[1], s + 1), (A[2], s + l), (A[3], s - l), (A[4], s - 1)])) for s ∈ S⁺])
		for (action, edge) ∈ zip(A, [range(5, 25, 5), range(21, 25), range(1, 5), range(1, 21, 5)])
			
			for loc ∈ edge
			
				mapping[Int(loc)][action] = Int(loc)
				
			end
			
		end
		
		return new(l, size, mapping, A, S⁺)

	end

end

mutable struct CornerWorld
	params::MazeGrid
	size
	mapping
	A
	S⁺
	S
	R
	function setWalls(grid::MazeGrid, walls)
	
		for wall in walls
			for i in grid.S⁺
				for action in ["↓", "←", "↑", "→"]
					for brick in wall
						if grid.mapping[i][action] == brick
							grid.mapping[i][action] = i
						end
					end
				end
			end
		end
		return grid.mapping
	end
	
	function CornerWorld(params, walls, goal)
		size = params.size
		mapping = params.mapping
		A = params.A
		S⁺ = params.S⁺
		mapping = setWalls(params, walls)
		S = [s for s in params.S⁺ if s ∉ Set(vcat(walls...))] 
		R = Dict([(s,0.0) for s in S])
		R[goal] = 10.0
		return new(params, size, mapping, A, S⁺, S, R)
	end
end

function dyna_q(game::CornerWorld, goal, α, γ)
	Q = Dict(s => Dict(a => rand() for a in game.A) for s in game.S)
	Model = Dict(s => Dict(a => [] for a in game.A) for s in game.S)
	buffer = Dict(s => Set() for s in game.S)

	s = rand([s for s in game.S if s != goal])

	
	for i in 1:500
		a = eGreedy(Q[s], 0.05)
		s′ = game.mapping[s][a]
		r = game.R[s′]
		Q[s][a] = Q[s][a] + α * (r + γ * maximum(collect(values(Q[s′]))) - Q[s][a])
		push!(buffer[s], a)
		Model[s][a] = [s′, r]
		s = s′

		for j in 1:50
			s_p, a_p = [(v, rand(collect(buffer[v]))) for v in keys(buffer) if !isempty(collect(buffer[v]))] |> rand

			s′_p, r_p = Model[s_p][a_p]
			
			Q[s_p][a_p] = Q[s_p][a_p] + α * (r_p + γ * maximum(collect(values(Q[s′_p]))) - Q[s_p][a_p])	
		end

	end
	return Q
end



function agentPlay(game::CornerWorld, goal, agent)
	trajectory = Tuple{Int, String, Int, Float64, Bool}[]
	
	s = rand([s for s in game.S if s != goal])
	t = false
	i = 0
	
	while t != true && i < 50
		a = collect(keys(agent[s]))[argmax(collect(values(agent[s])))]
		s′ = game.mapping[s][a]
		if s′ == goal
			t = true
		end
		push!(trajectory, tuple(s, a, s′, game.R[s′], t))
		s = s′
		i += 1
	end
	return trajectory
end #agentPlay

function randomPlay(game::CornerWorld, goal)
	trajectory = Tuple{Int, String, Int, Float64, Bool}[]
	
	s = rand([s for s in game.S if s != goal])
	t = false
	i = 0
	
	while t != true && i < 50
		a = rand(game.A)
		s′ = game.mapping[s][a]
		if s′ == goal
			t = true
		end
		push!(trajectory, tuple(s, a, s′, game.R[s′], t))
		s = s′
		i += 1
	end
	return trajectory
end #randomTrajectory


reshape(1:25, (5,5))'


walls = [[7, 8],  [17, 22], [9, 14, 19]]
goal = 25
g = CornerWorld(MazeGrid(5), walls, goal)
q_agent = dyna_q(g, goal, 0.3, 0.9)

maze = zeros(25)
for el ∈ Set(vcat(walls...))
	maze[el] = 1.0
end
maze[goal] = 0.5


Gray.(reshape(maze, (5,5))')

totalRandomReward = []
for i in 1:50
	push!(totalRandomReward, randomPlay(g, goal)[end][4])
end
println("Average Reward: $((totalRandomReward |> sum) / 50)")
plot(totalRandomReward |> cumsum)
plot!(totalRandomReward)


totalAgentReward = []
for i in 1:50
	push!(totalAgentReward, agentPlay(g, goal, q_agent)[end][4])
end
println("Average Reward: $((totalAgentReward |> sum) / 50)")
plot(totalAgentReward |> cumsum)
plot!(totalAgentReward)

