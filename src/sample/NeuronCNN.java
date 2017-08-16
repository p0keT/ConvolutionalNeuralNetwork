package sample;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

public class NeuronCNN {

    private ArrayList<WeightsCNN> weightsCNN = new ArrayList<>();
    private int learning_rate;
    private ArrayList<int[][]> input = new ArrayList<>();


    NeuronCNN(int[][] input, WeightsCNN weightsCNN, int learning_rate){
        this.weightsCNN.add(weightsCNN);
        this.learning_rate = learning_rate;
        this.input.add(input);
    }

    /**
     * Пряме розповсюдження помилки.
     */
    public void predict(){

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

        ArrayList<int[][]> results = new ArrayList<>();


        for (int i = 0; i < weights.size(); i++) {
            results.add(new int[height_of_input-(height_of_weight-1)][width_of_input-(width_of_weight-1)]);
            //j/k координаты левого/нижнего угла маски (массива весов) на изображении(прошлом слое)
            for (int j = height_of_weight-1; j < height_of_input; j++) {
                for (int k = width_of_weight-1; k < width_of_input; k++) {
                    //l/m координаты елементов маски (массива весов) на изображении(прошлом слое)
                    for (int l = j-(height_of_weight-1); l < height_of_weight; l++) {
                        for (int m = k-(width_of_weight-1); m <width_of_weight ; m++) {
                            inputPart[yTemp][xTemp]=input[l][m];
                            xTemp++;
                        }
                        yTemp++;
                    }
                    xTemp=0;
                    yTemp=0;
                    results.get(results.size()-1)[j][k]=convMath(inputPart,weights.get(i));
                }
            }
        }

        this.input = results;
        
    }

    /**
     * Слой RELU (блок линейной ректификации) применяет поэлементную
     * функцию активации вроде f (x) = max(0,x), устанавливая нулевой порог.
     * Иными словами, RELU выполняет следующие действия: если x > 0,
     * то объем остается прежним, а если x < 0,
     * то осекаются ненужные детали в канале и путем замены на 0.
     */
    public void reLU(){

    }

    /**
     * Слой FC (полносвязный слой) выводит N-мерный вектор (N — число классов)
     * для определения нужного класса(самолет, машына, кот). Работа организуется путем обращения к выходу
     * предыдущего слоя (карте признаков) и определения свойств,
     * которые наиболее характерны для определенного класса.
     */
    public void fC(){

    }


    public int convMath(int[][] inputPart, int[][] weight)
    {
        int[][] mul = new int[weight.length][weight[0].length];
        for (int i = 0; i <weight.length; i++)
        {
            for (int j = 0; j <weight[0].length; j++)
            {
                mul[i][ j] = new Color(inputPart[i][j]).getRed()*weight[i][j];
                mul[i][ j] += new Color(inputPart[i][j]).getBlue()*weight[i][j];
                mul[i][ j] += new Color(inputPart[i][j]).getGreen()*weight[i][j];
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
