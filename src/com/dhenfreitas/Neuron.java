package com.dhenfreitas;

public abstract class Neuron {
    protected double[] weights;
    protected double threshold;

    public Neuron() {

    }

    public Neuron(double threshold, int nweights) {
        // initializating weights
        this.threshold = threshold;
        this.weights = new double[nweights];
        for (int i = 0; i < nweights; i++) {
            this.weights[i] = Math.random() * 2 - 1;
        }
    }

    public abstract double process(double[] input) throws Exception;

    public double activationFunction(double net) {
        if (net > threshold)
            return 1.0;
        return 0.0;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double[] getWeights() {
        return this.weights;
    }
}
