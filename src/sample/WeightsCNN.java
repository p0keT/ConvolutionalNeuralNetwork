package sample;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class WeightsCNN {

    private ArrayList<double[][]> weights = new ArrayList<double[][]>();
    private String path = "";

    public WeightsCNN(int numb_of_weights, int size_of_weight, int rand){
        for (int i = 0; i < numb_of_weights; i++) {
            weights.add(new double[size_of_weight][size_of_weight]);

            for (int j = 0; j <size_of_weight ; j++) {
                for (int k = 0; k <size_of_weight ; k++) {
                    weights.get(weights.size()-1)[j][k] =Math.round(new Random().nextInt(5)); //Math.round(new Random().nextDouble()*100)/100.0;
                }
            }
        }
    }

    public void setWeights(int index, double[][] newWeight){
        this.weights.set(index,newWeight);
    }

    public WeightsCNN(ArrayList<double[][]> weights){
        this.weights = weights;
    }

    public WeightsCNN(String path){
        //реалізувати пізніше
    }

    public void save() throws IOException {
        //реалізувати пізніше
    }

    public ArrayList<double[][]> getWeights(){
        return weights;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

}
