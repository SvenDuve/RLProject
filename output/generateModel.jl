using Pkg   

Pkg.activate(".")


using RLTypes
using NODEDynamics
using BSON: @save


lunarlander_node_model = modelEnv(LunarLander(), ModelParameter(collect_train=1000, collect_test=100, training_episodes=500, batch_size=512, hidden=10, model_η=0.0005))

@save "lunarlander_node_model.bson" lunarlander_node_model # done