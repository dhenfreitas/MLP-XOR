package com.dhenfreitas;

public class NeuronIdentity extends Neuron {

    public NeuronIdentity() {
        super();
    }

    public double process(double[] input) throws Exception {
        if (input.length != 1) {
            throw new Exception("Input vector is incorrect.");
        }

        return input[0];
    }
}

