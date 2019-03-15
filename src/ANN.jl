module ANN

# types
export ANNLayer, ANNetwork

# functions
export newlayer, newnetwork, feedforward, feedforwardlayer, newnetworkwith
export identity, sigmoid

include("network.jl")

end