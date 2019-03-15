struct ANNLayer
    ω
    b
    σ
    dσ
end

struct ANNetwork
    layers::Array{ANNLayer, 1}
end

sigmoid = x -> 1 / (1 + exp(-x))

function newlayer(inputsize, layersize, activation, derivative)
    ANNLayer(
        rand(Float64, layersize, inputsize),
        rand(Float64, layersize),
        activation,
        derivative)
end

function newnetwork(layerssize, activation, derivative, 
                    outputactivation, outputderivative)
    layernumber = size(layerssize)[1] - 1
    layers = map(2:layernumber) do i
        newlayer(layerssize[i-1], layerssize[i], activation, derivative)
    end
    push!(layers, newlayer(layerssize[end-1], layerssize[end], activation, derivative))
    ANNetwork(layers)
end

function newnetworkwith(weights, activation, derivative, outputactivation, outputderivative, bias=true)
    
    layers = map(weights[1:end-1]) do i
        ANNLayer(
            i,
            bias ? ones(size(i)[1]) : zeros(size(i)[1]),
            activation,
            derivative
        )
    end
    b = bias ? ones(size(weights[end])[1]) : zeros(size(weights[end])[1])
    push!(layers, ANNLayer(weights[end], b, outputactivation, outputderivative))
    ANNetwork(layers)
end

function feedforwardlayer(layer, a)
    map(layer.ω * a + layer.b) do z
        layer.σ(z)
    end
end

function feedforward(network, input)
    a = Array[input,]
    for (i, e) in enumerate(network.layers)
        push!(a, feedforwardlayer(e, a[i]))
    end
    a
end