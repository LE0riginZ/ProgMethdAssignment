#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> //for generating random seed

#define HEIGHT 100
#define WIDTH 10

void populateData(FILE *input, double data[][10]);
void printData(double data[][10]);
void initializeWeight(double weight[10]);
void feedForward(double data[][10], double weight[10], double trainingOutput[90]);
void feedBack(double data[][10], double weight[10], double trainingOutput[90]);
double calcMMSE(double data[][10], double trainingOutput[90]);
double calcMAE(double data[][10], double trainingOutput[90]);
//Include arrays for list of MAE and MMSEs

int main()
{
    srand(time(NULL));                        //random seed for rand() generated
    FILE *infile_ptr;                         //initialize file pointer
    double data[100][10], trainingOutput[90]; //initialize 2D array for data set and training output
    double MAE = 0, MMSE = 0;
    double weight[10];
    int iteration = 0;
    infile_ptr = fopen("fertility_Diagnosis_Data_Group1_4.txt", "r"); //open file

    if (infile_ptr == NULL) //error checking in case file is not found
    {
        printf("File could not be found :(");
        exit(1);
    }

    populateData(infile_ptr, data); //call function to read file into 2D array
                                    //printData(data);                //call function to print data

    if (iteration == 0)
    {
        initializeWeight(weight);                  //fill weight array with random numbers from -1<=x<=1
        feedForward(data, weight, trainingOutput); //test run, not part of final
        MAE = calcMAE(data, trainingOutput);       //test run, not part of final
        iteration++;
        printf("\n---------------------------------------------\n");
    }
   /* while(MAE>0.25){
        feedBack(data, weight, trainingOutput);
        feedForward(data, weight, trainingOutput);
        MAE=calcMAE(data, trainingOutput);
        iteration++;
    }*/

    for(int i=0;i<10;i++){
         feedBack(data, weight, trainingOutput);
        feedForward(data, weight, trainingOutput);
        MAE=calcMAE(data, trainingOutput);
        iteration++;
        //printf("\n***********************************************\nIteration: %d\n***********************************************",iteration);
    }

    //printf("\nTotal iterations: %d",iteration);
    fclose(infile_ptr); //close file
}

//FUNCTIONS
void populateData(FILE *input, double data[][10])
{
    //read file data into 2D array
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            fscanf(input, "%lf,", &data[i][j]);
        }
    }
}

void printData(double data[][10])
{
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            printf("%.2f ", data[i][j]);
        }
        printf("\n"); //print results to check
    }
}

void initializeWeight(double weight[10])
{
    for (int i = 0; i < 10; i++)
    {
        weight[i] = rand() % 3 - 1;
        //printf("\n%lf", weight[i]);
    }
}

void feedForward(double data[][10], double weight[10], double trainingOutput[90])
{
    for (int i = 0; i < 90; i++)
    {
        double z = 0;
        for (int j = 0; j < 9; j++)
        {
            double w = weight[j];
            double x = data[i][j];
            double b = weight[10];
            double sigmoid = 0;
            z += (x * w) + b;
        }
        double sigmoid = 1 / (1 + exp((double)-z));
        trainingOutput[i] = sigmoid;
        //printf("%lf",sigmoid);
    }
}

void feedBack(double data[][10], double weight[10], double trainingOutput[90])
{
   double newCalc=0;
   double nWeight;
   double nBias;
   for(int i=0;i<90;i++){
       double y=trainingOutput[i];
       double d=data[i][10];
       double z = 0;
        for (int j = 0; j < 9; j++)
        {
            double w = weight[j];
            double x = data[i][j];
            double b = weight[10];
            double sigmoid = 0;
            z += (x * w) + b;
        }
        newCalc+=(y-d)*(exp(z)/(pow(1+exp(z),2)));
   }
   newCalc=newCalc/90;
   for(int j=0;j<9;j++){
       nWeight=weight[j]-newCalc*weight[j];
       weight[j]=nWeight;
       //printf("%lf\n",weight[j]);
   }
   nBias=weight[10]-newCalc;
   weight[10]=nBias;
   //printf("%lf\n",weight[10]);
}

double calcMMSE(double data[][10], double trainingOutput[90])
{
    double summation = 0;
    double MMSE = 0;
    for (int i = 0; i < 90; i++)
    {
        summation += pow(trainingOutput[i] - data[i][10], 2);
    }
    MMSE = summation / 90;
    printf("MMSE: %lf", MMSE);

    return MMSE;
}
double calcMAE(double data[][10], double trainingOutput[90])
{
    double summation = 0;
    double MAE = 0;
    for (int i = 0; i < 90; i++)
    {
        summation += abs(trainingOutput[i] - data[i][10]);
    }
    MAE = summation / 90;
    printf("\nMAE: %lf", MAE);
    return MAE;
}
