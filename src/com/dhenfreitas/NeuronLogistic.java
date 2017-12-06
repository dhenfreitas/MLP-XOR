package com.dhenfreitas;

public class NeuronLogistic extends Neuron {
    public NeuronLogistic(double threshold, int nweights) {
        super(threshold, nweights);
    }

    public double process(double[] input) throws Exception {
        // net = input[0] * weights[0] +
        // 	 input[1] * weights[1] +
        // 	 input[2] * weights[2] +
        // 	 ... + bias

        // weights = bias, weight_0, weight_1 ...
        double net = 1.0 * weights[0];
        for (int i = 1; i < weights.length; i++) {
            net += weights[i] * input[i-1];
        }

        return activationFunction(net);
    }

    public double activationFunction(double net) {
        return 1.0 / (1.0 + Math.exp(-net));
    }
}