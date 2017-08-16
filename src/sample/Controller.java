package sample;

import javafx.fxml.Initializable;

import java.net.URL;
import java.util.ArrayList;
import java.util.ResourceBundle;

public class Controller implements Initializable{

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        WeightsCNN w1 = new WeightsCNN(10,5,255);
        ArrayList<int[][]> weights = w1.getWeights();

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j <5 ; j++) {
                for (int k = 0; k <5 ; k++) {
                    System.out.print(" "+weights.get(i)[j][k]);
                }
                System.out.println();
            }
            System.out.println();
        }
    }
}
