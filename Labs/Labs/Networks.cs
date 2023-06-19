using System;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Labs
{
    /// <summary>
    /// Многослойный перцептрон
    /// </summary>
    internal class MLP
    {
        private class Neuron
        {
            internal float v;   // значение
            internal float[] w; // веса (от текущего к правому)
            internal float[] dw; // велечина поправки веса на предыдущем шаге
            internal float e;   // ошибка

            internal Neuron()
            {

            }

            internal void Sigmoid()
            {
                v = 1.0f / (1.0f + (float)Math.Exp(-v));
            }

            internal void ReLU()
            {
                if (v >= 0)
                    v = 1;
                else
                    v = 0;
            }
        }

        private Neuron[][] layers; // содержит входные скрытые и выходные сло
        private float a = 0.7f; // скорость обучения
        private float y = 0.8f;

        internal MLP(params int[] layerSizes)
        {
            layers = new Neuron[layerSizes.Length][];
            Random rand = new Random();

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Neuron[layerSizes[i]];

                for (int j = 0; j < layers[i].Length; j++)
                {
                    layers[i][j] = new Neuron();

                    if (i != layers.Length - 1)
                    {
                        layers[i][j].w = new float[layerSizes[i + 1]];
                        layers[i][j].dw = new float[layerSizes[i + 1]];

                        for (int k = 0; k < layers[i][j].w.Length; k++)
                            layers[i][j].w[k] = (float)rand.NextDouble();
                    }
                }
            }
        }

        internal float[] Test(params float[] input)
        {
            // переписываем данные во входной слой
            for (int i = 0; i < layers[0].Length; i++)
                layers[0][i].v = input[i];

            // считаем выход
            for (int i = 1; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i].Length; j++)
                {
                    layers[i][j].v = 0;

                    for (int k = 0; k < layers[i - 1].Length; k++)
                        layers[i][j].v += layers[i - 1][k].w[j] * layers[i - 1][k].v;

                    layers[i][j].Sigmoid();
                }
            }

            // нормализация
            float[] output = new float[layers[layers.Length - 1].Length];

            for (int i = 0; i < output.Length; i++)
                output[i] = layers[layers.Length - 1][i].v;


            return output;
        }

        int count; // количество эпох пройденных при обучении
        float c = 0; // максимальная ошибка выходного слоя

        internal void Train(float[][] input, float[][] real)
        {
            while (true)
            {
                c = 0;
                for (int i = 0; i < input.Length; i++)
                    Train(input[i], real[i]);

                count += 1;
                if (c < 0.0001f)
                {
                    Console.WriteLine("Количество эпох: " + count);
                    break;
                }
            }
        }

        private void Train(float[] input, float[] real)
        {
            float[] imitate; // результат прямого обхода
            float e; // буфер

            // прямой обход
            imitate = Test(input);

            //  E = 0.5*Math.Pow((expected[i] - output[i]),2);
            //  E = (expected[i] - output[i]);
            // считаем ошибку для выходного нейрона
            for (int i = 0; i < layers[layers.Length - 1].Length; i++)
            {
                layers[layers.Length - 1][i].e = (imitate[i] - real[i]) * layers[layers.Length - 1][i].v * (1f - layers[layers.Length - 1][i].v);
                c = Math.Max(c, layers[layers.Length - 1][i].e); //Math.Abs
            }

            // считаем ошибку скрытых слоёв
            // слои
            for (int i = layers.Length - 2; i > 0; i--)
            {
                // нейроны
                for (int j = 0; j < layers[i].Length; j++)
                {
                    e = 0;

                    // веса
                    for (int k = 0; k < layers[i][j].w.Length; k++)
                        e += layers[i][j].w[k] * layers[i + 1][k].e;

                    layers[i][j].e = e * layers[i][j].v * (1f - layers[i][j].v);
                }
            }


            // обновляем веса
            // слои
            for (int i = layers.Length - 2; i >= 0; i--)
            {
                // нейроны
                for (int j = 0; j < layers[i].Length; j++)
                {
                    // веса
                    for (int k = 0; k < layers[i][j].w.Length; k++)
                    {
                        layers[i][j].dw[k] = y * layers[i][j].dw[k] - a * layers[i + 1][k].e * layers[i][j].v;
                        layers[i][j].w[k] += layers[i][j].dw[k];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Самоорганизующиеся карты Кохонена
    /// </summary>
    internal class SOM
    {
        private float[,,] w; // веса

        // положение узла в сетке с минимальным расстоянием
        private int x; 
        private int y;
        private float radius;
        float a = 0.5f; // скорость обучения

        internal SOM(int x,int y,int z)
        {
            w = new float[x, y, z];
            radius = Math.Max(x,y);

            Random rand = new Random();

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    for (int k = 0; k < z; k++)
                    {
                        w[i, j, k] = (float)rand.NextDouble();
                    }
                }
            }
        }

        internal float[] Test(params float[] input)
        {
            float min = float.MaxValue;
            float distance;

            for (int i = 0; i < w.GetLength(0); i++)
            {
                for (int j = 0; j < w.GetLength(1); j++)
                {
                    distance = 0;

                    for (int k = 0; k < w.GetLength(2); k++)
                    {
                        distance += (float)Math.Pow(input[k] - w[i,j,k], 2);
                    }

                    distance = (float)Math.Sqrt(distance);

                    if(min>distance)
                    {
                        min = distance;
                        x = i;
                        y = j;
                    }
                }
            }

            float[] output = new float[w.GetLength(2)];

            for(int i=0;i<output.Length;i++)
                output[i] = w[x, y, i];

            return output;
        }


        internal void Train(float[][]input,int numberEpochs)
        {
            for (int i = 0; i < numberEpochs; i++)
            {
                Console.WriteLine(i);
                for (int j = 0; j < input.Length; j++)
                {
                    Train(input[j], i+1, numberEpochs);
                }
            }
        }

        private void Train(float[] input, int i, int numberEpochs)
        {
            float[] output = Test(input);

            // вычислляем радиус
            float bmu = (1f - (i / (float)numberEpochs))*radius;

            // изменяем веса внутри радиуса
            for (int j = 0; j < w.GetLength(0); j++)
            {
                for (int k = 0; k < w.GetLength(1); k++)
                {
                    // проверка на вхождение в радиус
                    if (Math.Sqrt(Math.Pow(i - x, 2) + Math.Pow(j - y, 2)) <= bmu)
                    {
                        for (int q = 0; q < w.GetLength(2); q++)
                        {
                            w[j, k, q] +=  a * bmu * (input[q] - w[j, k, q]);

                            // применяем сигмоиду, иначе значения уйдут за границы 0:1
                            w[j, k, q] = 1.0f / (1.0f + (float)Math.Exp(-w[j, k, q]));
                        }
                    }
                }
            }

        }
    }

    /// <summary>
    /// Сеть Хопфилда
    /// </summary>
    internal class RNN
    {
        private float[,] w;

        internal RNN(int countNeurons)
        {
            w = new float[countNeurons, countNeurons];
        }

        internal float[] Test(params float[] input)
        {
            float[] output = new float[w.GetLength(0)];

            for (int i = 0; i < w.GetLength(0); i++)
            {
                for (int j = 0; j < w.GetLength(1); j++)
                {
                    output[i] += input[j]*w[i,j];
                }

                // функция активации
                if (output[i] > 0)
                    output[i] = 1;
                else if (output[i] < 0)
                    output[i] = -1;
            }
          

            return output; 
        }

        internal void Train(params float[] input)
        {
            for (int i = 0; i < w.GetLength(0); i++)
            {
                for (int j = 0; j < w.GetLength(1); j++)
                {
                    if (i != j)
                        w[i, j] += input[i] * input[j]; 
                }
            }
        }
    }

    /// <summary>
    /// Сеть радиально-базисных функций
    /// </summary>
    internal class RBF
    {
        private class Neuron
        {
            internal float v;   // значение
            internal float[] w; // веса (от текущего к правому)
            internal float[] c; // центроиды
            internal float[] dw; // велечина поправки веса на предыдущем шаге
            internal float e;   // ошибка

            internal Neuron()
            {

            }

            internal void Sigmoid()
            {
                v = 1.0f / (1.0f + (float)Math.Exp(-v));
            }

            internal void ReLU()
            {
                if (v >= 0)
                    v = 1;
                else
                    v = 0;
            }
        }

        private Neuron[][] layers; // содержит входные скрытые и выходные сло
        internal float r = 2.5f; // ширина (стандартное отклоение)
        private float a = 0.7f; // скорость обучения
        private float y = 0.2f;

        internal RBF(params int[] layerSizes)
        {
            if (layerSizes.Length != 3)
                throw new Exception();

            layers = new Neuron[layerSizes.Length][];
            Random rand = new Random();

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Neuron[layerSizes[i]];

                for (int j = 0; j < layers[i].Length; j++)
                {
                    layers[i][j] = new Neuron();

                    if (i != layers.Length - 1 && i!=0)
                    {
                        layers[i][j].w = new float[layerSizes[i + 1]];
                        layers[i][j].dw = new float[layerSizes[i + 1]];
                        layers[i][j].c = new float[layerSizes[0]];

                        // веса
                        for (int k = 0; k < layers[i][j].w.Length; k++)
                            layers[i][j].w[k] = (float)rand.NextDouble();

                        // центроиды
                        for (int k = 0; k < layers[i][j].c.Length; k++)
                            layers[i][j].c[k] = (float)rand.NextDouble();
                    }
                }
            }


        }

        internal float[] Test(params float[] input)
        {
            // переписываем данные во входной слой
            for (int i = 0; i < layers[0].Length; i++)
                layers[0][i].v = input[i];

            float d; // расстояние

            // считаем выход
            // слои
            for (int i = 1; i < layers.Length; i++)
            {
                // нейроны
                for (int j = 0; j < layers[i].Length; j++)
                {
                    // если не последний слой
                    if (i != layers.Length - 1)
                    {
                        d = 0;

                        for (int k = 0; k < layers[i][j].c.Length; k++)
                            d += (float)Math.Pow(layers[i - 1][k].v - layers[i][j].c[k], 2);

                        layers[i][j].v = (float)Math.Exp(-d / r);
                    }

                    // умножаем на веса следующего слоя
                    if (i > 1)
                    {
                        layers[i][j].v = 0;

                        // перебор нейронов предыдущего слоя
                        for (int k = 0; k < layers[i-1].Length; k++)
                            layers[i][j].v += layers[i-1][k].w[j] * layers[i - 1][k].v; 
                        
                    }
                    /* layers[i][j].v = 0;

                     for (int k = 0; k < layers[i - 1].Length; k++)
                         layers[i][j].v += layers[i - 1][k].w[j] * layers[i - 1][k].v;

                     layers[i][j].Sigmoid();*/
                    layers[i][j].Sigmoid();
                }
            }            


            // нормализация
            float[] output = new float[layers[layers.Length - 1].Length];

            for (int i = 0; i < output.Length; i++)
                output[i] = layers[layers.Length - 1][i].v;


            return output;
        }

        int count; // количество эпох пройденных при обучении
        float c = 0; // максимальная ошибка выходного слоя

        internal void Train(float[][] input, float[][] real)
        {
            while (true)
            {
                c = 0;
                for (int i = 0; i < input.Length; i++)
                    Train(input[i], real[i]);

                count += 1;
                Console.WriteLine(c);
                if (c < 0.2f)
                {
                    Console.WriteLine("Количество эпох: " + count);
                    break;
                }
            }
        }
    
        private void Train(float[] input, float[] real)
        {
            float[] imitate; // результат прямого обхода
            float e; // буфер

            // прямой обход
            imitate = Test(input);

            //  E = 0.5*Math.Pow((expected[i] - output[i]),2);
            //  E = (expected[i] - output[i]);
            // считаем ошибку для выходного нейрона
            for (int i = 0; i < layers[layers.Length - 1].Length; i++)
            {
                layers[layers.Length - 1][i].e = (imitate[i] - real[i]) * layers[layers.Length - 1][i].v * (1f - layers[layers.Length - 1][i].v);
                c = Math.Max(c, Math.Abs(imitate[i] - real[i])); // Math.Abs
            }

            //Console.WriteLine(imitate[0] + "   " + c);

            // считаем ошибку скрытых слоёв
            // слои
            for (int i = layers.Length - 2; i > 0; i--)
            {
                // нейроны
                for (int j = 0; j < layers[i].Length; j++)
                {
                    e = 0;

                    // веса
                    for (int k = 0; k < layers[i][j].w.Length; k++)
                        e += layers[i][j].w[k] * layers[i + 1][k].e;

                    layers[i][j].e = e * layers[i][j].v * (1f - layers[i][j].v);
                }
            }

            int minC;
            float minValue;

            // обновляем веса
            // слои
            for (int i = layers.Length - 2; i > 0; i--)
            {
                minC = 0;
                minValue = float.MaxValue;

                // нейроны
                for (int j = 0; j < layers[i].Length; j++)
                {
                    // веса
                    for (int k = 0; k < layers[i][j].w.Length; k++)
                    {
                        layers[i][j].dw[k] = y * layers[i][j].dw[k] - a * layers[i + 1][k].e * layers[i][j].v;
                        layers[i][j].w[k] += layers[i][j].dw[k];
                    }

                    // проверка на минимальное расстояние
                    if (layers[i][j].v < minValue)
                    {
                        minC = j;
                        minValue = layers[i][j].v;
                    }
                }
                // меняем центроид с минимальным расстоянием
                for(int k = 0; k < layers[i][minC].c.Length;k++)
                {
                    layers[i][minC].c[k] -= (layers[i][minC].c[k] - layers[i - 1][k].v);
                }
            }
        }       
    }

    /// <summary>
    /// Обобщенно-регрессионная нейронная сеть
    /// </summary>
    internal class GRNN
    {
        private class Neuron
        {
            internal float v;   // значение
            internal float[] w; // веса (от текущего к правому)
            internal float[] c; // центроиды
            internal float[] dw; // велечина поправки веса на предыдущем шаге
            internal float e;   // ошибка

            internal Neuron()
            {

            }

            internal void Sigmoid()
            {
                v = 1.0f / (1.0f + (float)Math.Exp(-v));
            }

            internal void ReLU()
            {
                if (v >= 0)
                    v = 1;
                else
                    v = 0;
            }
        }

        private Neuron[][] layers; // содержит входные скрытые и выходные сло
        internal float r = 2.5f; // ширина (стандартное отклоение)
        private float a = 0.7f; // скорость обучения
        private float y = 0.2f;

        internal GRNN(params int[] layerSizes)
        {
            if (layerSizes.Length != 4)
                throw new Exception();

            layers = new Neuron[layerSizes.Length][];
            Random rand = new Random();

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Neuron[layerSizes[i]];

                for (int j = 0; j < layers[i].Length; j++)
                {
                    layers[i][j] = new Neuron();

                    if (i != layers.Length - 1 && i != 0)
                    {
                        layers[i][j].w = new float[layerSizes[i + 1]];
                        layers[i][j].dw = new float[layerSizes[i + 1]];
                        layers[i][j].c = new float[layerSizes[0]];

                        // веса
                        for (int k = 0; k < layers[i][j].w.Length; k++)
                            layers[i][j].w[k] = (float)rand.NextDouble();

                        // центроиды
                        for (int k = 0; k < layers[i][j].c.Length; k++)
                            layers[i][j].c[k] = (float)rand.NextDouble();
                    }
                }
            }


        }

        internal float[] Test(params float[] input)
        {
            // переписываем данные во входной слой
            for (int i = 0; i < layers[0].Length; i++)
                layers[0][i].v = input[i];

            float d; // расстояние

            // считаем выход
            // слои
            for (int i = 1; i < layers.Length; i++)
            {
                // нейроны
                for (int j = 0; j < layers[i].Length; j++)
                {
                    // если не последний слой
                    if (i != layers.Length - 1)
                    {
                        d = 0;

                        for (int k = 0; k < layers[i][j].c.Length; k++)
                            d += (float)Math.Pow(layers[i - 1][k].v - layers[i][j].c[k], 2);

                        layers[i][j].v = (float)Math.Exp(-d / r);
                    }

                    // умножаем на веса следующего слоя
                    if (i > 1)
                    {
                        layers[i][j].v = 0;

                        // перебор нейронов предыдущего слоя
                        for (int k = 0; k < layers[i - 1].Length; k++)
                            layers[i][j].v += layers[i - 1][k].w[j] * layers[i - 1][k].v;

                    }
                    /* layers[i][j].v = 0;

                     for (int k = 0; k < layers[i - 1].Length; k++)
                         layers[i][j].v += layers[i - 1][k].w[j] * layers[i - 1][k].v;

                     layers[i][j].Sigmoid();*/
                    layers[i][j].Sigmoid();
                }
            }


            // нормализация
            float[] output = new float[layers[layers.Length - 1].Length];

            for (int i = 0; i < output.Length; i++)
                output[i] = layers[layers.Length - 1][i].v;


            return output;
        }

        int count; // количество эпох пройденных при обучении
        float c = 0; // максимальная ошибка выходного слоя

        internal void Train(float[][] input, float[][] real)
        {
            while (true)
            {
                c = 0;
                for (int i = 0; i < input.Length; i++)
                    Train(input[i], real[i]);

                count += 1;
                Console.WriteLine(c);
                if (c < 0.2f)
                {
                    Console.WriteLine("Количество эпох: " + count);
                    break;
                }
            }
        }

        private void Train(float[] input, float[] real)
        {
            float[] imitate; // результат прямого обхода
            float e; // буфер

            // прямой обход
            imitate = Test(input);

            //  E = 0.5*Math.Pow((expected[i] - output[i]),2);
            //  E = (expected[i] - output[i]);
            // считаем ошибку для выходного нейрона
            for (int i = 0; i < layers[layers.Length - 1].Length; i++)
            {
                layers[layers.Length - 1][i].e = (imitate[i] - real[i]) * layers[layers.Length - 1][i].v * (1f - layers[layers.Length - 1][i].v);
                c = Math.Max(c, Math.Abs(imitate[i] - real[i])); // Math.Abs
            }

            //Console.WriteLine(imitate[0] + "   " + c);

            // считаем ошибку скрытых слоёв
            // слои
            for (int i = layers.Length - 2; i > 0; i--)
            {
                // нейроны
                for (int j = 0; j < layers[i].Length; j++)
                {
                    e = 0;

                    // веса
                    for (int k = 0; k < layers[i][j].w.Length; k++)
                        e += layers[i][j].w[k] * layers[i + 1][k].e;

                    layers[i][j].e = e * layers[i][j].v * (1f - layers[i][j].v);
                }
            }

            int minC;
            float minValue;

            // обновляем веса
            // слои
            for (int i = layers.Length - 2; i > 0; i--)
            {
                minC = 0;
                minValue = float.MaxValue;

                // нейроны
                for (int j = 0; j < layers[i].Length; j++)
                {
                    // веса
                    for (int k = 0; k < layers[i][j].w.Length; k++)
                    {
                        layers[i][j].dw[k] = y * layers[i][j].dw[k] - a * layers[i + 1][k].e * layers[i][j].v;
                        layers[i][j].w[k] += layers[i][j].dw[k];
                    }

                    // проверка на минимальное расстояние
                    if (layers[i][j].v < minValue)
                    {
                        minC = j;
                        minValue = layers[i][j].v;
                    }
                }
                // меняем центроид с минимальным расстоянием
                for (int k = 0; k < layers[i][minC].c.Length; k++)
                {
                    layers[i][minC].c[k] -= (layers[i][minC].c[k] - layers[i - 1][k].v);
                }
            }
        }
    }

    /// <summary>
    /// Генетические алгоритмы
    /// </summary>
    internal class GA
    {
        private class Neuron
        {
            internal float v;   // значение
            internal float[] w; // веса (от текущего к правому)


            internal Neuron()
            {

            }

            internal void Sigmoid()
            {
                v = 1.0f / (1.0f + (float)Math.Exp(-v));
            }

            internal void ReLU()
            {
                if (v >= 0)
                    v = 1;
                else
                    v = 0;
            }
        }

        private Neuron[][] n; // выбранная особь
        private Neuron[][] n1; // особь 1
        private Neuron[][] n2; // особь 2
        private float a = 0.1f; // скорость мутации

        internal GA(params int[] layerSizes)
        {
            n1 = new Neuron[layerSizes.Length][];
            InitN(n1,layerSizes);
            n2 = new Neuron[layerSizes.Length][];
            InitN(n2, layerSizes);
        }

        private void InitN(Neuron[][] layers, params int[] layerSizes)
        {
            Random rand = new Random();

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Neuron[layerSizes[i]];

                for (int j = 0; j < layers[i].Length; j++)
                {
                    layers[i][j] = new Neuron();

                    if (i != layers.Length - 1)
                    {
                        layers[i][j].w = new float[layerSizes[i + 1]];

                        for (int k = 0; k < layers[i][j].w.Length; k++)
                            layers[i][j].w[k] = (float)rand.NextDouble();
                    }
                }
            }
        }

        internal float[] Test(params float[] input)
        {
            // переписываем данные во входной слой
            for (int i = 0; i < n[0].Length; i++)
                n[0][i].v = input[i];

            // считаем выход
            for (int i = 1; i < n.Length; i++)
            {
                for (int j = 0; j < n[i].Length; j++)
                {
                    n[i][j].v = 0;

                    for (int k = 0; k < n[i - 1].Length; k++)
                        n[i][j].v += n[i - 1][k].w[j] * n[i - 1][k].v;

                    n[i][j].Sigmoid();
                }
            }

            // нормализация
            float[] output = new float[n[n.Length - 1].Length];

            for (int i = 0; i < output.Length; i++)
                output[i] = n[n.Length - 1][i].v;


            return output;
        }

        internal void Train(float[][] input, float[][] real)
        {
            float[] imitate;
            float c1; // ошибка первой особи
            float c2; // ошибка второй особи

            while (true)
            {
                c1 = 0;
                c2 = 0;

                n = n1;
                // проводим вычисления первой особи
                for(int i=0;i<input.Length;i++)
                {
                    imitate = Test(input[i]);

                    for (int j = 0; j < imitate.Length; j++)
                        c1 += Math.Abs(imitate[j] - real[i][j]);
                }

                n = n2;
                // проводим вычисления второй особи
                for (int i = 0; i < input.Length; i++)
                {
                    imitate = Test(input[i]);

                    for (int j = 0; j < imitate.Length; j++)
                        c2 += Math.Abs(imitate[j] - real[i][j]);
                }

                Console.WriteLine(c1 + " " + c2);
                // селекция
                if(c1< c2)
                {
                    // первая особь лучше
                    if (c1 < 0.0001f)
                    {
                        n = n1;
                        break;
                    }
                    UpdateW(n2,n1);
                }
                else
                {
                    // вторая особь лучше
                    if (c2 < 0.0001f)
                    {
                        n = n2;
                        break;
                    }
                    UpdateW(n1,n2);
                }
            }
        }

        private void UpdateW(Neuron[][] layers1, Neuron[][] layers2)
        {
            Random rand = new Random();

            for(int i=0;i< layers1.Length-1;i++)
            {
                for(int j = 0; j < layers1[i].Length;j++)
                {
                    for(int k = 0; k < layers1[i][j].w.Length;k++)
                    {
                        layers1[i][j].w[k] = layers2[i][j].w[k];
                        layers1[i][j].w[k] += ((float)rand.NextDouble()-0.5f)*a;
                    }
                }
            }
        }
    }
}