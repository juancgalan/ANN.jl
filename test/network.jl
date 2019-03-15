using Test

@testset "FeedForward" begin

@testset "Creation of a single neuron and two inputs" begin
    ans = newlayer(2, 1, identity, identity)
    @test ans.dσ == identity
    @test ans.σ == identity
    @test size(ans.b) == (1,)
    @test size(ans.ω) == (1,2)
end

@testset "Creation of a 5 neuron layer with 11 inputs" begin
    ans = newlayer(11, 5, identity, identity)
    @test size(ans.b) == (5,)
    @test size(ans.ω) == (5, 11)
end

@testset "Creation of a network with 3, 2, 1" begin
    ans = newnetwork([3, 2, 1], identity, identity, identity, identity)
    @test size(ans.layers) == (2,)
    layer1 = ans.layers[1]
    @test size(layer1.b) == (2,)
    @test size(layer1.ω) == (2, 3)
    layer2 = ans.layers[2]
    @test size(layer2.b) == (1,)
    @test size(layer2.ω) == (1, 2)
end

@testset "Feedforward layer XOR with 1, 1 one step" begin
    weights = [0.8 0.2
               0.4 0.9
               0.3 0.5]
    bias = [0; 0; 0]
    expected = [0.73105857863
                0.78583498304
                0.68997448112]
    layer = ANNLayer(weights, bias, x -> 1/(1 + exp(-x)), x -> x)
    m = feedforwardlayer(layer, [1; 1])
    @test sqrt(sum((m- expected).^2)) <= 0.0001
end

@testset "Feedforward 3, 1 XOR with 1, 1 one step" begin
    weights = Array[[0.8 0.2; 0.4 0.9; 0.3 0.5], [0.3 0.5 0.9]]
    net = newnetworkwith(weights, sigmoid, identity, sigmoid, identity, false)
    m = feedforward(net, [1; 1])
    expected = Array[[0.73105857863; 0.78583498304; 0.68997448112], [0.7746924929149283]]
    for (i,e) in zip(m[2:end], expected)
        @test sqrt(sum((i - e).^2)) <= 0.001
    end
end

end

nothing