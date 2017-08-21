package sample;

import javafx.fxml.Initializable;

import java.awt.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.ResourceBundle;

public class Controller implements Initializable{

    @Override
    public void initialize(URL location, ResourceBundle resources) {
//        WeightsCNN w1 = new WeightsCNN(10,5,1000);
//        ArrayList<double[][]> weights = w1.getWeights();
//
//        for (int i = 0; i < 1; i++) {
//            for (int j = 0; j <5 ; j++) {
//                for (int k = 0; k <5 ; k++) {
//                    System.out.print(" "+weights.get(i)[j][k]);
//                }
//                System.out.println();
//            }
//            System.out.println();
//        }
//        RWFile rw = new RWFile();
//        double[][] temp2 = rw.readBufferedArrayImage("(1).jpg");
//        for (int j = 0; j <50 ; j++) {
//            for (int k = 0; k <50 ; k++) {
//                System.out.print("("+new Color((int) temp2[j][k]).getRed()+"/"+new Color((int) temp2[j][k]).getGreen()+"/"+new Color((int) temp2[j][k]).getBlue()+")");
//            }
//            System.out.println();
//        }
//        System.out.println();
//        NeuronCNN cnn  = new NeuronCNN(temp2,w1,1000);
//        ArrayList<double[][]> temp = cnn.getLastLayerArrs();
//        for (int j = 0; j <45 ; j++) {
//            for (int k = 0; k <45 ; k++) {
//                System.out.print(" "+temp.get(0)[j][k]);
//            }
//            System.out.println();
//        }
//        System.out.println();
//        cnn.reLU();
//
//        temp = cnn.getLastLayerArrs();
//        for (int j = 0; j <45 ; j++) {
//
//            for (int k = 0; k <45 ; k++) {
//                System.out.print(" "+temp.get(temp.size()-1)[j][k]);
//            }
//            System.out.println();
//        }
//
//        ArrayList<LayerCNN> layers = cnn.getLayersCNN();
//        for (int i = 0; i < layers.size(); i++) {
//            System.out.println("+"+layers.get(i).getLayer().size()+"+"+layers.get(i).getLayer().get(0).length+"/"+layers.get(i).getLayer().get(0)[0].length);
//        }
//        cnn.fC();
        trainTest();

    }

    public void trainTest(){
        WeightsCNN w1 = new WeightsCNN(5,5,1000);
        RWFile rw = new RWFile();
        NeuronCNN cnn  = new NeuronCNN("E:\\JProjects\\ConvolutionalNeuralNetwork\\neural",w1,0.5);
        cnn.predict();
        ArrayList<double[][]> weights0 = cnn.getWeightsCNN().get(6).getWeights();
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j <5 ; j++) {
                for (int k = 0; k <5 ; k++) {
                    System.out.print(" "+weights0.get(i)[j][k]);
                }
                System.out.println();
            }
            System.out.println();
        }
        cnn.train(10);



        ArrayList<double[][]> weights = cnn.getWeightsCNN().get(6).getWeights();
        System.out.println(cnn.getWeightsCNN().size());
        System.out.println("----------------------------------------------");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j <5 ; j++) {
                for (int k = 0; k <5 ; k++) {
                    if(weights.get(i)[j][k]==weights0.get(i)[j][k])
                        System.out.print(" "+weights.get(i)[j][k]);
                    else
                        System.out.print(" -");
                }
                System.out.println();
            }
            System.out.println();
        }
        System.out.println();

        cnn.changeInput(0);
        cnn.predict(cnn.getWeightsCNN().size());
    }
}
