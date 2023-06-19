using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Labs
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //XOR
            /*    MLP p = new MLP(2, 3, 3, 1);

                float[][] samples = new float[][]
                {
                   new float[]{0,0 },
                   new float[]{0,1 },
                   new float[]{1,0 },
                   new float[]{1,1 },
                };

                float[][] real = new float[][]
                {
                   new float[]{0 },
                   new float[]{1 },
                   new float[]{1 },
                   new float[]{0 },
                };

                p.Train(samples,real);

                Console.WriteLine(p.Test(0, 0)[0]);
                Console.WriteLine(p.Test(0, 1)[0]);
                Console.WriteLine(p.Test(1, 0)[0]);
                Console.WriteLine(p.Test(1, 1)[0]);*/


            //NUMBERS
            /* MLP p = new MLP(15, 10);

             float[][] samples = new float[][]
             {
               new float[]{1,1,1,1,0,0,1,1,1,0,0,1,1,1,1 },
               new float[]{0,0,1,0,1,0,1,0,1,0,0,1,0,0,1 },
               new float[]{1,1,1,0,0,1,1,1,1,1,0,0,1,1,1 },
               new float[]{1,1,1,0,0,1,1,1,1,0,0,1,1,1,1 },
             };

             float[][] real = new float[][]
             {
                new float[]{1,0,0,0,0,0,0,0,0,0 },
                new float[]{0,1,0,0,0,0,0,0,0,0 },
                new float[]{0,0,1,0,0,0,0,0,0,0 },
                new float[]{0,0,0,1,0,0,0,0,0,0 },
             };

             p.Train(samples, real);


             for(int i=0;i<10;i++)
             {
                 Console.WriteLine(p.Test(new float[] { 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1 })[i]);
             }
            */


            //COLORS
            /* SOM p = new SOM(400,400,3);


             p.Train(new float[][]
             {
                 new float [] {1.0f,0.0f,0.0f }, // красный
                 new float [] {0.0f,0.5f,0.0f }, // зеленый
                 new float [] {0.0f,0.0f,1.0f }, // голубой
                 new float [] {0.0f,0.39f,0.0f }, // темно-зеленый
                 new float [] {0.0f,0.0f,0.54f }, // темно-синий
                 new float [] {1.0f,1.0f,0.0f }, // желтый
                 new float [] {1.0f,0.64f,0.0f }, // оранжевый
                 new float [] {0.5f,0.0f,0.5f }, // фиолетовый
             },10);


             float[] r;

             r = p.Test(0.1f, 0.5f, 0.6f);
             Console.Write(r[0] + " " + r[1] + " " + r[2]);
            */


            //IMAGE
            /* RNN p = new RNN(4);

             p.Train(1.0f,1.0f,-1.0f,-1.0f);

             float[] r = p.Test(1.0f, 1.0f, -1.0f, -1.0f);

             for (int i = 0; i < r.Length; i++)
                 Console.Write(r[i]+" ");
            */


            //APPROXIMATION MLP
            /* MLP p = new MLP(1,5,5,1);

             p.Train(
                 new float[][]
                 {
                     new float[]{0.0f},
                     new float[]{0.25f},
                     new float[]{0.5f},
                     new float[]{0.75f},
                     new float[]{1.0f},
                 },
                 new float[][] 
                 {
                     new float[]{(float)Math.Sin(0.0f) },
                     new float[]{(float)Math.Sin(0.25f) },
                     new float[]{(float)Math.Sin(0.5f) },
                     new float[]{(float)Math.Sin(0.75f) },
                     new float[]{(float)Math.Sin(1.0f) },
                 }
                 );


             Console.WriteLine(Math.Sin(0.1f) + "  " + p.Test(new float[] { 0.1f })[0]);
             Console.WriteLine(Math.Sin(0.5f) + "  " + p.Test(new float[] { 0.5f })[0]);
             Console.WriteLine(Math.Sin(0.6f) + "  " + p.Test(new float[] { 0.6f })[0]);
             Console.WriteLine(Math.Sin(0.77f) + "  " + p.Test(new float[] { 0.77f })[0]);*/


            //APPROXIMATION RBF
            GRNN p = new GRNN(1,10,15,1);

            p.Train(new float[][]
   {
new float[]{0.02f},
new float[]{0.04f},
new float[]{0.06f},
new float[]{0.08f},
new float[]{0.1f},
new float[]{0.12f},
new float[]{0.14f},
new float[]{0.16f},
new float[]{0.18f},
new float[]{0.2f},
new float[]{0.22f},
new float[]{0.24f},
new float[]{0.26f},
new float[]{0.28f},
new float[]{0.3f},
new float[]{0.32f},
new float[]{0.34f},
new float[]{0.36f},
new float[]{0.38f},
new float[]{0.4f},
new float[]{0.42f},
new float[]{0.44f},
new float[]{0.46f},
new float[]{0.48f},
new float[]{0.5f},
new float[]{0.52f},
new float[]{0.54f},
new float[]{0.56f},
new float[]{0.58f},
new float[]{0.6f},
new float[]{0.62f},
new float[]{0.64f},
new float[]{0.66f},
new float[]{0.68f},
new float[]{0.7f},
new float[]{0.72f},
new float[]{0.74f},
new float[]{0.76f},
new float[]{0.78f},
new float[]{0.8f},
new float[]{0.82f},
new float[]{0.84f},
new float[]{0.86f},
new float[]{0.88f},


   }, new float[][]
   {
new float[]{(float)Math.Sin(0.02f) },
new float[]{(float)Math.Sin(0.04f) },
new float[]{(float)Math.Sin(0.06f) },
new float[]{(float)Math.Sin(0.08f) },
new float[]{(float)Math.Sin(0.1f) },
new float[]{(float)Math.Sin(0.12f) },
new float[]{(float)Math.Sin(0.14f) },
new float[]{(float)Math.Sin(0.16f) },
new float[]{(float)Math.Sin(0.18f) },
new float[]{(float)Math.Sin(0.2f) },
new float[]{(float)Math.Sin(0.22f) },
new float[]{(float)Math.Sin(0.24f) },
new float[]{(float)Math.Sin(0.26f) },
new float[]{(float)Math.Sin(0.28f) },
new float[]{(float)Math.Sin(0.3f) },
new float[]{(float)Math.Sin(0.32f) },
new float[]{(float)Math.Sin(0.34f) },
new float[]{(float)Math.Sin(0.36f) },
new float[]{(float)Math.Sin(0.38f) },
new float[]{(float)Math.Sin(0.4f) },
new float[]{(float)Math.Sin(0.42f) },
new float[]{(float)Math.Sin(0.44f) },
new float[]{(float)Math.Sin(0.46f) },
new float[]{(float)Math.Sin(0.48f) },
new float[]{(float)Math.Sin(0.5f) },
new float[]{(float)Math.Sin(0.52f) },
new float[]{(float)Math.Sin(0.54f) },
new float[]{(float)Math.Sin(0.56f) },
new float[]{(float)Math.Sin(0.58f) },
new float[]{(float)Math.Sin(0.6f) },
new float[]{(float)Math.Sin(0.62f) },
new float[]{(float)Math.Sin(0.64f) },
new float[]{(float)Math.Sin(0.66f) },
new float[]{(float)Math.Sin(0.68f) },
new float[]{(float)Math.Sin(0.7f) },
new float[]{(float)Math.Sin(0.72f) },
new float[]{(float)Math.Sin(0.74f) },
new float[]{(float)Math.Sin(0.76f) },
new float[]{(float)Math.Sin(0.78f) },
new float[]{(float)Math.Sin(0.8f) },
new float[]{(float)Math.Sin(0.82f) },
new float[]{(float)Math.Sin(0.84f) },
new float[]{(float)Math.Sin(0.86f) },
new float[]{(float)Math.Sin(0.88f) },


   });


            Console.WriteLine(Math.Sin(0.9f) + " " + p.Test(new float[] { 0.9f })[0]);
            Console.WriteLine(Math.Sin(0.92f) + " " + p.Test(new float[] { 0.92f })[0]);
            Console.WriteLine(Math.Sin(0.94f) + " " + p.Test(new float[] { 0.94f })[0]);
            Console.WriteLine(Math.Sin(0.96f) + " " + p.Test(new float[] { 0.96f })[0]);
            Console.WriteLine(Math.Sin(0.98f) + " " + p.Test(new float[] { 0.98f })[0]);
            Console.WriteLine(Math.Sin(1.0f) + " " + p.Test(new float[] { 1.0f })[0]);
        


        //SEQUENCE
        /*GRNN p = new GRNN(1, 15,15, 1);

        p.Train(new float[][]
        {
             new float[]{0.0f},
             new float[]{0.1f},
             new float[]{0.2f},
             new float[]{0.3f},
             new float[]{0.4f},
             new float[]{0.5f},
        }, new float[][]
        {
            new float[]{(float)Math.Sin(0.0f) },
            new float[]{(float)Math.Sin(0.1f) },
            new float[]{(float)Math.Sin(0.2f) },
            new float[]{(float)Math.Sin(0.3f) },
            new float[]{(float)Math.Sin(0.4f) },
            new float[]{(float)Math.Sin(0.5f) },
        });


        Console.WriteLine(Math.Sin(0.6f) + "  " + p.Test(new float[] { 0.6f })[0]);
        Console.WriteLine(Math.Sin(0.7f) + "  " + p.Test(new float[] { 0.7f })[0]);
        Console.WriteLine(Math.Sin(0.8f) + "  " + p.Test(new float[] { 0.8f })[0]);
        Console.WriteLine(Math.Sin(0.9f) + "  " + p.Test(new float[] { 0.9f })[0]);*/


        //GENETIC
        /*GA p = new GA(2, 3, 3, 1);

        float[][] samples = new float[][]
        {
               new float[]{0,0 },
               new float[]{0,1 },
               new float[]{1,0 },
               new float[]{1,1 },
        };

        float[][] real = new float[][]
        {
               new float[]{0 },
               new float[]{1 },
               new float[]{1 },
               new float[]{0 },
        };

        p.Train(samples, real);

        Console.WriteLine(p.Test(0, 0)[0]);
        Console.WriteLine(p.Test(0, 1)[0]);
        Console.WriteLine(p.Test(1, 0)[0]);
        Console.WriteLine(p.Test(1, 1)[0]);*/



        Console.ReadKey();
        }
    }
}
