package com.dhenfreitas;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Network {

    private List<Neuron> inputLayer;
    private List<Neuron> hiddenLayer;
    private List<Neuron> outputLayer;
    private String trainFilename;
    private String testFilename;
    private Scanner trainFile;
    private Scanner testFile;
    private double alpha;
    private final double THRESHOLD = 0.5;
    private final double MAX_ERROR = 0.05;

    public Network(int inputLayerSize, int hiddenLayerSize,
                   int outputLayerSize, String trainFilename,
                   String testFilename, double alpha)
            throws Exception {

        this.alpha = alpha;
        this.trainFilename = trainFilename;
        this.testFilename = testFilename;

        this.inputLayer = new ArrayList<Neuron>();
        this.hiddenLayer = new ArrayList<Neuron>();
        this.outputLayer = new ArrayList<Neuron>();

        for (int i = 0 ; i < inputLayerSize; i++)
            this.inputLayer.add(
                    new NeuronIdentity()
            );
        for (int i = 0; i < hiddenLayerSize; i++)
            this.hiddenLayer.add(
                    new NeuronLogistic(THRESHOLD,
                            inputLayerSize+1)
            );
        for (int i = 0; i < outputLayerSize; i++)
            this.outputLayer.add(
                    new NeuronLogistic(THRESHOLD,
                            hiddenLayerSize+1)
            );
    }

    public void train() throws Exception {
        double accumulatedError = 0.0;
        int iterations = 0;
        System.out.println("Inputs\tOutputs\tObtained");

        do {

            this.trainFile = new Scanner(
                    new FileReader(trainFilename)
            );

            // ler um arquivo de entrada
            while (this.trainFile.hasNext()) {

                iterations++;

                ////////////////////////////////////////////
                // leitura do proximo exemplo de treinamento
                ////////////////////////////////////////////

                // lendo um exemplo do arquivo de treinamento
                double input[] = new double[inputLayer.size()];
                for (int i = 0; i < inputLayer.size(); i++) {
                    input[i] = this.trainFile.nextDouble();
                    System.out.print(input[i]+"\t");
                }

                System.out.print("|\t");

                // lendo a saida esperada para o exemplo
                double output[]=new double[outputLayer.size()];
                for (int i = 0; i < outputLayer.size(); i++) {
                    output[i] = this.trainFile.nextDouble();
                    System.out.print(output[i]+"\t");
                }
                System.out.print("|\t");

                ////////////////////////////////////
                // processamento da rede
                ////////////////////////////////////
                // camada de entrada
                double inputIL2HL[] = new double[
                        inputLayer.size()
                        ];
                for (int i = 0; i < inputLayer.size(); i++) {
                    Neuron neuron = inputLayer.get(i);
                    inputIL2HL[i] =
                            neuron.process(
                                    new double[] { input[i] } );
                }

                // camada escondida
                double inputHL2OL[] = new double[
                        hiddenLayer.size()
                        ];
                for (int i = 0; i < hiddenLayer.size(); i++) {
                    Neuron neuron = hiddenLayer.get(i);
                    inputHL2OL[i] =
                            neuron.process(inputIL2HL);
                }

                // camada de saida
                double obtained[] = new double [
                        outputLayer.size()
                        ];
                for (int i = 0; i < outputLayer.size(); i++) {
                    Neuron neuron = outputLayer.get(i);
                    obtained[i] =
                            neuron.process(inputHL2OL);
                    System.out.print(obtained[i]);
                }
                System.out.println();

                //////////////////////////////////////////
                // treinamento da rede
                //////////////////////////////////////////

                // output -> o que espero da rede
                // obtained -> o que rede gerou de fato
                //
                // erro = output - obtained

                // CAMADA DE SAIDA
                double bpOL2HL[] =
                        new double[hiddenLayer.size()];

                for (int i = 0; i < hiddenLayer.size(); i++)
                    bpOL2HL[i] = 0.0;

                for (int i = 0; i < output.length; i++) {
                    // erro para o neuronio i da
                    // 	camada de saida
                    double delta_o =
                            (output[i] - obtained[i]) *
                                    obtained[i]*(1.0-obtained[i]);

                    accumulatedError +=
                            Math.pow(output[i]-obtained[i],2.0);

                    Neuron neuron = outputLayer.get(i);
                    double[] weights = neuron.getWeights();

                    // bias
                    weights[0] = weights[0] + alpha *
                            delta_o * 1.0;

                    for (int j=1; j < weights.length;j++) {
                        weights[j] = weights[j] +
                                alpha * delta_o *
                                        inputHL2OL[j-1];

                        bpOL2HL[j-1]
                                += delta_o * weights[j];
                    }

                    neuron.setWeights(weights);
                }

                // CAMADA ESCONDIDA
                for (int i = 0; i < hiddenLayer.size(); i++) {
                    Neuron neuron = hiddenLayer.get(i);
                    double weights[] = neuron.getWeights();

                    double delta_h = inputHL2OL[i] *
                            (1.0-inputHL2OL[i]) *
                            bpOL2HL[i];

                    // bias
                    weights[0] = weights[0] + alpha *
                            delta_h * 1.0;

                    for (int j=1; j<weights.length;j++) {
                        weights[j] = weights[j] +
                                alpha * delta_h *
                                        inputIL2HL[j-1];
                    }

                    neuron.setWeights(weights);
                }
            }

            this.trainFile.close();

            System.out.println("accumulatedError: "+
                    (accumulatedError/(iterations*1.0)));

        } while (accumulatedError/(iterations*1.0) > MAX_ERROR);
    }

    public void test () throws Exception
    {

        Scanner testFile = new Scanner (new FileReader (this.testFilename));

        System.out.
                println ("Input\t\t|\tExpected Output\t\t|\tObtained Output");

        while (testFile.hasNext ())
        {
            // Reading training data
            double input[] = new double[inputLayer.size ()];
            for (int i = 0; i < inputLayer.size (); i++)
            {
                input[i] = testFile.nextDouble ();
                System.out.print (input[i] + "\t");
            }

            System.out.print ("|\t");

            // expected output
            double expectedOutput[] = new double[outputLayer.size ()];
            for (int i = 0; i < outputLayer.size (); i++)
            {
                expectedOutput[i] = testFile.nextDouble ();
                System.out.print (expectedOutput[i] + "\t");
            }

            // processing input layer
            double inputIL2HL[] = new double[inputLayer.size ()];
            for (int i = 0; i < inputLayer.size (); i++)
            {
                Neuron neuron = inputLayer.get (i);
                inputIL2HL[i] = neuron.process (new double[]
                        {
                                input[i]}
                );
            }

            // processing hidden layer
            double inputHL2OL[] = new double[hiddenLayer.size ()];
            for (int i = 0; i < hiddenLayer.size (); i++)
            {
                Neuron neuron = hiddenLayer.get (i);
                inputHL2OL[i] =	// i_p
                        neuron.process (inputIL2HL);
            }

            System.out.print ("|\t");

            // processing output layer
            double obtainedOutput[] = new double[outputLayer.size ()];
            for (int i = 0; i < outputLayer.size (); i++)
            {
                Neuron neuron = outputLayer.get (i);
                obtainedOutput[i] = neuron.process (inputHL2OL);
                System.out.print (obtainedOutput[i] + "\t");
            }
            System.out.println ();
        }
    }

    public static void main(String args[]) throws Exception {
//        if (args.length != 6) {
//            System.out.println("usage: java Network inputLayerSize hiddenLayerSize outputLayerSize alpha trainFile.dat testFile.dat");
//            System.exit(0);
//        }
//
//        int inputLayerSize = Integer.parseInt(args[0]);
//        int hiddenLayerSize = Integer.parseInt(args[1]);
//        int outputLayerSize = Integer.parseInt(args[2]);
//        double alpha =  Double.parseDouble(args[3]);
//        String trainFilename = args[4];
//        String testFilename = args[5];

        int inputLayerSize = 2;
        int hiddenLayerSize = 3;
        int outputLayerSize = 2;
        double alpha =  0.01;
        String curDir = System.getProperty("user.dir");
        String trainFilename = curDir+"/test/xor-test.dat";
        String testFilename = curDir+"/train/xor-train.dat";

        Network network = new Network(
                inputLayerSize, hiddenLayerSize,
                outputLayerSize, trainFilename,
                testFilename, alpha
        );
        network.train();
        network.test();
    }
}
