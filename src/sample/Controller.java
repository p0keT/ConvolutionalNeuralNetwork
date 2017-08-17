package sample;

import javafx.fxml.Initializable;

import java.awt.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.ResourceBundle;

public class Controller implements Initializable{

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        WeightsCNN w1 = new WeightsCNN(10,5,10);
        ArrayList<int[][]> weights = w1.getWeights();

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j <5 ; j++) {
                for (int k = 0; k <5 ; k++) {
                    System.out.print(" "+weights.get(i)[j][k]);
                }
                System.out.println();
            }
            System.out.println();
        }
        RWFile rw = new RWFile();
        int[][] temp2 = rw.readBufferedArrayImage("(1).jpg");
        for (int j = 0; j <50 ; j++) {
            for (int k = 0; k <50 ; k++) {
                System.out.print("("+new Color(temp2[j][k]).getRed()+"/"+new Color(temp2[j][k]).getGreen()+"/"+new Color(temp2[j][k]).getBlue()+")");
            }
            System.out.println();
        }
        System.out.println();
        NeuronCNN cnn  = new NeuronCNN(temp2,w1,1000);
        ArrayList<int[][]> temp = cnn.getLastLayerArrs();
        for (int j = 0; j <45 ; j++) {
            for (int k = 0; k <45 ; k++) {
                System.out.print(" "+temp.get(0)[j][k]);
            }
            System.out.println();
        }
        System.out.println();
        cnn.reLU();
        cnn.predict();
        cnn.reLU();
        cnn.predict();
        cnn.reLU();
        cnn.predict();
        cnn.reLU();
        temp = cnn.getLastLayerArrs();
        for (int j = 0; j <30 ; j++) {

            for (int k = 0; k <30 ; k++) {
                System.out.print(" "+temp.get(temp.size()-1)[j][k]);
            }
            System.out.println();
        }

        ArrayList<LayerCNN> layers = cnn.getLayerCNNs();
        for (int i = 0; i < layers.size(); i++) {
            System.out.println("+"+layers.get(i).getLayer().size()+"+"+layers.get(i).getLayer().get(0).length+"/"+layers.get(i).getLayer().get(0)[0].length);
        }
    }
}
