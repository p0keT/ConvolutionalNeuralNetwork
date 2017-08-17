package sample;

import java.io.IOException;
import java.util.ArrayList;

public class LayerCNN {

    int[][] input;
    private ArrayList<int[][]> layer = new ArrayList<int[][]>();
    private String path = "";

    LayerCNN(String path){
        this.path = path;
        input = new RWFile().readBufferedArrayImage(path);
    }

    LayerCNN(ArrayList<int[][]> layer){
        this.layer = layer;
    }

    LayerCNN(int[][] layer){
        this.layer.add(layer);
    }
    LayerCNN(){

    }

    public void addMap(int[][] map){
        layer.add(map);
    }

    public int[][] getInput() {
        return input;
    }

    public ArrayList<int[][]> getLayer() {
        return layer;
    }

    public String getPath() {
        return path;
    }
}
