package sample;

import java.util.ArrayList;

public class LayerCNN {

    double[][] input;
    private ArrayList<double[][]> layer = new ArrayList<double[][]>();
    private String path = "";

    LayerCNN(String path){
        this.path = path;
        input = new RWFile().readBufferedArrayImage(path);
    }

    LayerCNN(ArrayList<double[][]> layer){
        this.layer = layer;
    }

    LayerCNN(double[][] layer){
        this.layer.add(layer);
    }
    LayerCNN(){

    }

    public void addMap(double[][] map){
        layer.add(map);
    }

    public double[][] getInput() {
        return input;
    }

    public ArrayList<double[][]> getLayer() {
        return layer;
    }

    public String getPath() {
        return path;
    }
}
