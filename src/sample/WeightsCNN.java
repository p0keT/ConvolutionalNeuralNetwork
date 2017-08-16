package sample;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class WeightsCNN {

    private ArrayList<int[][]> weights = new ArrayList<>();
    private String path = "";

    public WeightsCNN(int numb_of_weights, int size_of_weight, int rand){
        for (int i = 0; i < numb_of_weights; i++) {
            weights.add(new int[size_of_weight][size_of_weight]);

            for (int j = 0; j <size_of_weight ; j++) {
                for (int k = 0; k <size_of_weight ; k++) {
                    weights.get(weights.size()-1)[j][k] = new Random().nextInt(rand);
                }
            }
        }
    }

    public WeightsCNN(String path){
        //реалізувати пізніше
    }

    public void save() throws IOException {
        //реалізувати пізніше
    }

    public ArrayList<int[][]> getWeights(){
        return weights;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

}
