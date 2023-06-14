using Pkg   

Pkg.activate(".")


using RLTypes
# using NODEDynamics
using ODERNNDynamics
using BSON: @save


lunarlanderdiscrete_odernn_model = modelEnv(LunarLanderDiscrete(), ModelParameter(collect_train=1000, collect_test=100, training_episodes=500, batch_size=512, hidden=10, model_η=0.0005))

@save "lunarlanderdiscrete_odernn_model.bson" lunarlanderdiscrete_odernn_model # done