package sample;

import java.awt.*;
import java.io.*;
import java.util.ArrayList;

public class NeuronCNN {

    public ArrayList<WeightsCNN> getWeightsCNN() {
        return weightsCNN;
    }

    private ArrayList<WeightsCNN> weightsCNN = new ArrayList<>();
    private ArrayList<WeightsCNN> newWeightsCNN = new ArrayList<>();
    private double answerFC=0.0;

    ArrayList<double[][]> images = new ArrayList<>();
    ArrayList<String> names = new ArrayList<>();

    private ArrayList<LayerCNN> layersCNN = new ArrayList<>();
    private double learning_rate;
    private int epochs;
    private double expected_error;

    public ArrayList<LayerCNN> getLayersCNN() {
        return layersCNN;
    }
    public ArrayList<double[][]> getLastLayerArrs() {
        return layersCNN.get(layersCNN.size()-1).getLayer();
    }

    public double getAnswerFC() {
        return answerFC;
    }


    NeuronCNN(String path, WeightsCNN weightsCNN, double learning_rate){
        File[] fList;
        File F = new File(path);
        fList = F.listFiles();

        for(int i=0; i<fList.length; i++)
        {
            if(fList[i].isFile()) {
                images.add(new RWFile().readBufferedArrayImage(path+"\\" + fList[i].getName()));
                names.add(fList[i].getName());

            }
        }
        //--------------------------------------------
        //this.weightsCNN.add(weightsCNN);
        this.learning_rate = learning_rate;
        //ArrayList<int[][]> input = new ArrayList<>();
        layersCNN.add(new LayerCNN(images.get(0)));

        //conV(input,weightsCNN.getWeights());
    }

    public void changeInput(int input){
        layersCNN.removeAll(layersCNN);
        layersCNN.add(new LayerCNN(images.get(input)));
        this.temp=false;
    }

    /**
     * Пряме розповсюдження помилки.
     * Викликається коли ваги стягуються з файлу
     */
    public void predict(int numbOfWeingtsCNN){
        for (int i = 0; i <numbOfWeingtsCNN ; i++) {
            conV(layersCNN.get(layersCNN.size() - 1).getLayer(), weightsCNN.get(i).getWeights());
            reLU();
        }
        weightsCNN.add(new WeightsCNN(5,5,1000));
        fC(weightsCNN);

    }

    /**
     * Пряме розповсюдження помилки.
     * Також створює новий набір вагів.
     * Якщо ваги стягуються з файлу викликайте перегрузку.
     */
    public void predict(){
        weightsCNN.add(new WeightsCNN(5, 5, 1000));
        conV(layersCNN.get(layersCNN.size() - 1).getLayer().get(0), weightsCNN.get(weightsCNN.size() - 1).getWeights());
        reLU();
        for (int i = 0; i <5 ; i++) {
            weightsCNN.add(new WeightsCNN(5, 5, 1000));
            conV(layersCNN.get(layersCNN.size() - 1).getLayer(), weightsCNN.get(weightsCNN.size() - 1).getWeights());
            reLU();
        }
        weightsCNN.add(new WeightsCNN(5,5,1000));
        fC(weightsCNN);

    }

    /**
     * Зворотнє розповсюдження помилки.
     * @param expected_predict - очікувана помилка.
     */
    public void train(double expected_predict){
        this.expected_error = expected_predict;
        for (int i = weightsCNN.size()-1; i >= 0; i--) {
            backPropagation(weightsCNN.get(i), layersCNN.get(i + 1));
        }
    }

    private void backPropagation(WeightsCNN weightsCNN, LayerCNN layer){
        double error;
        double weight_delta;
        double[][] tempWeights = new double[weightsCNN.getWeights().get(0).length][weightsCNN.getWeights().get(0)[0].length];


        for (int m = 0; m <layer.getLayer().size() ; m++) {
            for (int i = 0; i < weightsCNN.getWeights().get(0).length; i++) {
                for (int j = 0; j < weightsCNN.getWeights().get(0)[0].length; j++) {
                    tempWeights[i][j]=0;
                }
            }
            //i/j - індекси, що проходять по слоям
            for (int i = 0; i < layer.getLayer().get(m).length; i++) {
                for (int j = 0; j <layer.getLayer().get(m)[0].length; j++) {
                    double actual_error = layer.getLayer().get(m)[i][j];
                    //tempWeights.add(new double[weightsCNN.getWeights().get(0).length][weightsCNN.getWeights().get(0)[0].length]);
                    //k/l - індекси, що проходять по вагам
                    for (int k = 0; k <weightsCNN.getWeights().get(m).length; k++) {
                        for (int l = 0; l <weightsCNN.getWeights().get(m)[0].length; l++) {
                            error = actual_error - expected_error;
                            weight_delta = error*(actual_error*(1-actual_error));
                            tempWeights[k][l]+=weightsCNN.getWeights().get(m)[k][l]-weight_delta*learning_rate;
                            tempWeights[k][l]=Math.round(tempWeights[k][l]*100)/100.0;
                            tempWeights[k][l]=Math.round(tempWeights[k][l]*100)/100.0;
                            //System.out.print("+"+tempWeights[k][l]);
                        }
                        //System.out.println();
                    }
                    //System.out.println();

                }
            }
            for (int i = 0; i < weightsCNN.getWeights().get(0).length; i++) {
                for (int j = 0; j < weightsCNN.getWeights().get(0)[0].length; j++) {
                    tempWeights[i][j]/=layer.getLayer().get(m).length*layer.getLayer().get(m)[0].length;
                    tempWeights[i][j]=Math.round(tempWeights[i][j]*100)/100.0;
                }
                //System.out.println();
            }
            //System.out.println();

            weightsCNN.getWeights().set(m,tempWeights);
        }

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
    public void conV(double[][] input, ArrayList<double[][]> weights){
        //список весов на даном слое

        int width_of_weight = weights.get(0)[0].length;
        int height_of_weight = weights.get(0).length;
        int height_of_input = input.length;
        int width_of_input = input[0].length;
        double[][] inputPart = new double[height_of_weight][width_of_weight];
        int xTemp = 0;
        int yTemp = 0;

        ArrayList<double[][]> layerMaps = new ArrayList<>();


        for (int i = 0; i < weights.size(); i++) {
            layerMaps.add(new double[height_of_input-(height_of_weight-1)][width_of_input-(width_of_weight-1)]);
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

        this.temp=true;
        layersCNN.add(new LayerCNN(layerMaps));
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
    public void conV(ArrayList<double[][]> input, ArrayList<double[][]> weights){
        layersCNN.add(new LayerCNN());
        //список весов на даном слое
        for (int n = 0; n < input.size(); n++) {

            int width_of_weight = weights.get(0)[0].length;
            int height_of_weight = weights.get(0).length;
            int height_of_input = input.get(n).length;
            int width_of_input = input.get(n)[0].length;
            double[][] inputPart = new double[height_of_weight][width_of_weight];
            int xTemp = 0;
            int yTemp = 0;

            ArrayList<double[][]> layerMaps = new ArrayList<>();

                layerMaps.add(new double[height_of_input - (height_of_weight - 1)][width_of_input - (width_of_weight - 1)]);
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



            layersCNN.get(layersCNN.size()-1).addMap(layerMaps.get(0));
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
        ArrayList<double[][]> lastLayer = layersCNN.get(layersCNN.size()-1).getLayer();
//        for (int i = 0; i < lastLayer.size(); i++) {
//            for (int j = 0; j <lastLayer.get(i).length ; j++) {
//                for (int k = 0; k < lastLayer.get(i)[0].length; k++) {
//                    if(lastLayer.get(i)[j][k]<200){
//                        lastLayer.get(i)[j][k]=0;
//                    }
//
//                }
//            }
//        }
        for (int i = 0; i < lastLayer.size(); i++) {
            for (int j = 0; j <lastLayer.get(i).length ; j++) {
                for (int k = 0; k < lastLayer.get(i)[0].length; k++) {

                    lastLayer.get(i)[j][k]= (1/(1-Math.exp(lastLayer.get(i)[j][k]*-1)));
                    lastLayer.get(i)[j][k]= Math.round(lastLayer.get(i)[j][k]*100)/100.0;

                }
            }
        }
    }

    /**
     * Слой FC (полносвязный слой) выводит N-мерный вектор (N — число классов)
     * для определения нужного класса(самолет, машына, кот). Работа организуется путем обращения к выходу
     * предыдущего слоя (карте признаков) и определения свойств,
     * которые наиболее характерны для определенного класса.
     */
    public void fC(ArrayList<WeightsCNN> weightsCNN){
        ArrayList<double[][]> lastLayer = layersCNN.get(layersCNN.size()-1).getLayer();

        double[][] results = new double[lastLayer.size()][1];

        for (int i = 0; i < lastLayer.size(); i++) {
            results[i][0] = convMath(lastLayer.get(i),weightsCNN.get(weightsCNN.size()-1).getWeights().get(i));
            results[i][0]=Math.round(results[i][0]*100)/100.0;
            results[i][0]= (1/(1-Math.exp(results[i][0]*-1)));
            results[i][0]=Math.round(results[i][0]*100)/100.0;

        }

        for (int j = 0; j <results.length ; j++) {
            for (int k = 0; k < results[0].length; k++) {
                answerFC+=results[j][k];
            }
        }
        answerFC/=results.length;
        double[][] answer = new double[][] {{answerFC}};
        layersCNN.add(new LayerCNN(answer));
        System.out.println("*******************answer: "+answerFC);
    }

    //треба потім замінити на норм рішення
    private boolean temp=false;
    public double convMath(double[][] inputPart, double[][] weight)
    {
        double[][] mul = new double[weight.length][weight[0].length];
        if(temp==false) {
            for (int i = 0; i < weight.length; i++) {
                for (int j = 0; j < weight[0].length; j++) {
                    mul[i][j] = new Color((int) inputPart[i][j]).getRed() * weight[i][j];
                    mul[i][j] += new Color((int) inputPart[i][j]).getGreen() * weight[i][j];
                    mul[i][j] += new Color((int) inputPart[i][j]).getBlue() * weight[i][j];


                }
            }
        }
        else {
            for (int i = 0; i < weight.length; i++) {
                for (int j = 0; j < weight[0].length; j++) {
                    mul[i][j] = inputPart[i][j] * weight[i][j];
                }
            }
        }

        double sum = 0;
        for (int i = 0; i <weight.length; i++)
        {
            for (int j = 0; j <weight[0].length; j++)
            {
                sum += mul[i][ j];
            }
        }
        sum/=25;
        sum=Math.round(sum*100)/100.0;
        return sum;
    }

    public void save(){
        for (int i = 0; i <weightsCNN.size() ; i++) {
            for (int j = 0; j <weightsCNN.get(i).getWeights().size() ; j++) {
                FileWriter filewriter = null;
                new File("weights\\"+i+"layer").mkdirs();
                try {
                    filewriter = new FileWriter(new File("weights\\"+i+"layer\\"+j+".txt"));
                for (int k = 0; k <weightsCNN.get(i).getWeights().get(j).length ; k++) {
                    for (int l = 0; l <weightsCNN.get(i).getWeights().get(j)[0].length  ; l++) {
                        filewriter.write(weightsCNN.get(i).getWeights().get(j)[k][l]+"\n");
                    }
                    filewriter.write("+\n");
                }
                filewriter.flush();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }


    /**
     * зчитує дані з файлу
     * @param path петь к файлу
     * @return повертає список слів
     */
    public void readFile(String path)throws IOException{
        ArrayList<ArrayList<String>> s = new ArrayList<>();

        String str;

        BufferedReader in = null;
        in = new BufferedReader(new FileReader(path));


        int height =0;
        int width =0;
        s.add(new ArrayList<>());
        while ((str=in.readLine())!=null) {
            if(str.equals("+")) {
                height++;
                s.add(new ArrayList<>());
            }else {
                s.get(s.size()-1).add(str);
                width++;
            }
        }

        width=width/height;

        //weight = new int[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                //weight[i][j]=Integer.parseInt(s.get(i).get(j));
            }
        }


    }

}
