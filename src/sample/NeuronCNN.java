package sample;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

public class NeuronCNN {

    private ArrayList<WeightsCNN> weightsCNN = new ArrayList<>();

    public ArrayList<LayerCNN> getLayerCNNs() {
        return layerCNNs;
    }
    public ArrayList<int[][]> getLastLayerArrs() {
        return layerCNNs.get(layerCNNs.size()-1).getLayer();
    }


    private ArrayList<LayerCNN> layerCNNs = new ArrayList<>();
    private int learning_rate;

    NeuronCNN(int[][] input, WeightsCNN weightsCNN, int learning_rate){
        this.weightsCNN.add(weightsCNN);
        this.learning_rate = learning_rate;
        //ArrayList<int[][]> input = new ArrayList<>();
        layerCNNs.add(new LayerCNN(input));

        conV(input,weightsCNN.getWeights());
    }

    /**
     * Пряме розповсюдження помилки.
     * тут будемо в циклі викликати метод conV  для кожного input
     */
    public void predict(){
        weightsCNN.add(new WeightsCNN(10,5,10));

        conV(layerCNNs.get(layerCNNs.size()-1).getLayer(),weightsCNN.get(weightsCNN.size()-1).getWeights());

    }

    /**
     * Зворотнє розповсюдження помилки.
     * @param expected_predict - очікувана помилка.
     */
    public void train(int expected_predict){

    }
    
    /**
     * Слой CONV (слой свёртки) умножает значения фильтра
     * на исходные значения пикселей изображения (поэлементное умножение),
     * после чего все эти умножения суммируются.
     * Каждая уникальная позиция введенного изображения производит число.
     *
     * !!!короч при створенні екземпляру класу в конструкторі викличеться цей метод і передасться
     * картинка і список вагів. При наступних викликах метода(вручну) сюди буде передаватися один з нейронів слою і
     * один масив вагів або список цих масивів якщо ми захочемо збільшити кількість нейронів на наступному слої.
     * Як зменшити кількість нейронів на наступному слої я поки не вїхав.
     */
    public void conV(int[][] input, ArrayList<int[][]> weights){
        //список весов на даном слое

        int width_of_weight = weights.get(0)[0].length;
        int height_of_weight = weights.get(0).length;
        int height_of_input = input.length;
        int width_of_input = input[0].length;
        int[][] inputPart = new int[height_of_weight][width_of_weight];
        int xTemp = 0;
        int yTemp = 0;

        ArrayList<int[][]> layerMaps = new ArrayList<>();


        for (int i = 0; i < weights.size(); i++) {
            layerMaps.add(new int[height_of_input-(height_of_weight-1)][width_of_input-(width_of_weight-1)]);
            //j/k координаты левого/нижнего угла маски (массива весов) на изображении(прошлом слое)
            for (int j = height_of_weight-1; j < height_of_input; j++) {
                for (int k = width_of_weight-1; k < width_of_input; k++) {
                    //l/m координаты елементов маски (массива весов) на изображении(прошлом слое)
                    for (int l = j-(height_of_weight-1); l <= j; l++) {
                        for (int m = k-(width_of_weight-1); m <= k ; m++) {
                            inputPart[yTemp][xTemp]=input[l][m];
                            xTemp++;
                        }
                        yTemp++;
                        xTemp=0;
                    }

                    yTemp=0;
                    layerMaps.get(layerMaps.size()-1)[j-(height_of_weight-1)][k-(width_of_weight-1)]=convMath(inputPart,weights.get(i));
                }
            }
        }


        layerCNNs.add(new LayerCNN(layerMaps));
    }

    /**
     * Слой CONV (слой свёртки) умножает значения фильтра
     * на исходные значения пикселей изображения (поэлементное умножение),
     * после чего все эти умножения суммируются.
     * Каждая уникальная позиция введенного изображения производит число.
     *
     * !!!короч при створенні екземпляру класу в конструкторі викличеться цей метод і передасться
     * картинка і список вагів. При наступних викликах метода(вручну) сюди буде передаватися один з нейронів слою і
     * один масив вагів або список цих масивів якщо ми захочемо збільшити кількість нейронів на наступному слої.
     * Як зменшити кількість нейронів на наступному слої я поки не вїхав.
     */
    public void conV(ArrayList<int[][]> input, ArrayList<int[][]> weights){
        layerCNNs.add(new LayerCNN());
        //список весов на даном слое
        for (int n = 0; n < input.size(); n++) {

            int width_of_weight = weights.get(0)[0].length;
            int height_of_weight = weights.get(0).length;
            int height_of_input = input.get(n).length;
            int width_of_input = input.get(n)[0].length;
            int[][] inputPart = new int[height_of_weight][width_of_weight];
            int xTemp = 0;
            int yTemp = 0;

            ArrayList<int[][]> layerMaps = new ArrayList<>();

                layerMaps.add(new int[height_of_input - (height_of_weight - 1)][width_of_input - (width_of_weight - 1)]);
                //j/k координаты левого/нижнего угла маски (массива весов) на изображении(прошлом слое)
                for (int j = height_of_weight - 1; j < height_of_input; j++) {
                    for (int k = width_of_weight - 1; k < width_of_input; k++) {
                        //l/m координаты елементов маски (массива весов) на изображении(прошлом слое)
                        for (int l = j - (height_of_weight - 1); l <= j; l++) {
                            for (int m = k - (width_of_weight - 1); m <= k; m++) {
                                inputPart[yTemp][xTemp] = input.get(n)[l][m];
                                xTemp++;
                            }
                            yTemp++;
                            xTemp = 0;
                        }

                        yTemp = 0;
                        layerMaps.get(layerMaps.size() - 1)[j - (height_of_weight - 1)][k - (width_of_weight - 1)] = convMath(inputPart, weights.get(n));
                    }
                }



            layerCNNs.get(layerCNNs.size()-1).addMap(layerMaps.get(0));
        }
    }

    /**
     * Слой RELU (блок линейной ректификации) применяет поэлементную
     * функцию активации вроде f (x) = max(0,x), устанавливая нулевой порог.
     * Иными словами, RELU выполняет следующие действия: если x > 0,
     * то объем остается прежним, а если x < 0,
     * то осекаются ненужные детали в канале и путем замены на 0.
     *
     * !!!можливо активаційна функція буде інша (напр. сигмоїд)
     */
    public void reLU(){
        ArrayList<int[][]> lastLayer = layerCNNs.get(layerCNNs.size()-1).getLayer();
        for (int i = 0; i < lastLayer.size(); i++) {
            for (int j = 0; j <lastLayer.get(i).length ; j++) {
                for (int k = 0; k < lastLayer.get(i)[0].length; k++) {
                    if(lastLayer.get(i)[j][k]<20000){
                        lastLayer.get(i)[j][k]=0;
                    }
                }
            }
        }
//        for (int i = 0; i < input.size(); i++) {
//            for (int j = 0; j <input.get(i).length ; j++) {
//                for (int k = 0; k < input.get(i)[0].length; k++) {
//
//                        input.get(i)[j][k]= (int) (1/(1-Math.exp(input.get(i)[j][k]*-1)));
//
//                }
//            }
//        }
    }

    /**
     * Слой FC (полносвязный слой) выводит N-мерный вектор (N — число классов)
     * для определения нужного класса(самолет, машына, кот). Работа организуется путем обращения к выходу
     * предыдущего слоя (карте признаков) и определения свойств,
     * которые наиболее характерны для определенного класса.
     */
    public void fC(){
//        ArrayList<int[][]> lastInput = listInputs.get(listInputs.size()-1);
//
//        for (int i = 0; i <lastInput.size() ; i++) {
//
//        }
    }


    public int convMath(int[][] inputPart, int[][] weight)
    {
        int[][] mul = new int[weight.length][weight[0].length];
        for (int i = 0; i <weight.length; i++)
        {
            for (int j = 0; j <weight[0].length; j++)
            {
                mul[i][ j] = new Color(inputPart[i][j]).getRed()*weight[i][j];
                mul[i][ j] += new Color(inputPart[i][j]).getGreen()*weight[i][j];
                mul[i][ j] += new Color(inputPart[i][j]).getBlue()*weight[i][j];


            }
        }

        int sum = 0;
        for (int i = 0; i <weight.length; i++)
        {
            for (int j = 0; j <weight[0].length; j++)
            {
                sum += mul[i][ j];
            }
        }
        return sum;
    }

}
