#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> //for generating random seed and calculating time taken

#include "fileIn.h"    //File IO functions
#include "neuron.h"    //Neuron and Layers initialisation
#include "feed.h"      //feedForward and feedBack functions
#include "errorCalc.h" //Confusion Matrix, MAE, MMSE

void make2D(float **array, int row, int column);
void free2D(float **array, int row);
void freeArrays(float **data, float **trainingVar, float *trainingDes, float **testingVar, float *testingDes, float *trainingOutput, float *testingOutput, int dataRow, int trainingRow, int testingRow);
void initializeNetwork(Layer *network, int networkSize);
void printNetwork(Layer *network, int networkSize);
void freeNetwork(Layer *network, int networkSize);

int main()
{
    srand(time(NULL));                                                //seed for random number generator
    FILE *infile_ptr;                                                 //initialize file pointer
    infile_ptr = fopen("fertility_Diagnosis_Data_Group1_4.txt", "r"); //open file
    if (infile_ptr == NULL)                                           //exit if file is not found
    {
        printf("File could not be found :(");
        exit(1);
    }
    //////////////////////////////////////////////DATA SET PARAMETERS SETUP START///////////////////////////////////////////////////////////////////////////
    printf("----------------------------------------------------------------------------------------------------------\n");
    printf("DATA PARAMETERS SETTING\n");
    printf("----------------------------------------------------------------------------------------------------------\n");
    const int dataRow = countRow(infile_ptr);       //count number of rows in file
    const int dataColumn = countColumn(infile_ptr); //count number of columns in file
    printf("This file has %d rows and %d columns\n\n", dataRow, dataColumn);
    const int trainingPercent; //Get user input on % of data to use as training set
    printf("Percentage of data to be used as the training set (out of 100%%): ");
    scanf("%d", &trainingPercent);

    const int trainingRow = floor((dataRow * trainingPercent) / 100); //calc num of rows in training set
    const int testingRow = dataRow - trainingRow;                     // calc num of rows in testing set
    const int varColumn = dataColumn - 1;                             //training and testing variable arrays, does not include results column
    printf("\nNumber of rows in traning set: %d", trainingRow);
    printf("\nNumber of rows in testing set: %d\n", testingRow);

    float **data = (float **)malloc(dataRow * sizeof(*data));
    make2D(data, dataRow, dataColumn);                   // 2D array for data created
    populateData(infile_ptr, data, dataRow, dataColumn); //populate data into **data
    //printData(data, dataRow, dataColumn); //debugging

    //declare training set arrays
    float **trainingVar = (float **)calloc(trainingRow, sizeof(float *)); //training variables input
    make2D(trainingVar, trainingRow, varColumn);
    float *trainingDes = (float *)calloc(trainingRow, sizeof(float)); //desired training output
    //declare testing set arrays
    float **testingVar = (float **)calloc(testingRow, sizeof(float *)); //testing variables input
    make2D(testingVar, testingRow, varColumn);
    float *testingDes = (float *)calloc(testingRow, sizeof(float)); //desired testing output

    //creating arrays for output
    float *trainingOutput = (float *)calloc(trainingRow, sizeof(float));
    float *testingOutput = (float *)calloc(testingRow, sizeof(float));
    splitData(data, trainingVar, trainingDes, testingVar, testingDes, dataRow, dataColumn, trainingRow, varColumn); //split data set to different arrays

    //////////////////////////////////////////////DATA SET PARAMETERS SETUP END///////////////////////////////////////////////////////////////////////////*/

    //////////////////////////////////////////MACHINE LEARNING PARAMETERS SET UP START///////////////////////////////////////////////////////////////////
    printf("----------------------------------------------------------------------------------------------------------\n");
    printf("MACHINE LEARNING PARAMETERS SETTINGS\n");
    printf("----------------------------------------------------------------------------------------------------------\n");
    int hiddenLayerNum;
    float learningRate;
    float MAETarget;
    printf("Enter in Learning Rate: ");
    scanf("%f", &learningRate);
    printf("Enter in MAE Target: ");
    scanf("%f", &MAETarget);
    printf("\nNumber Of Hidden Layers: ");
    scanf("%d", &hiddenLayerNum);
    printf("----------------------------------------------------------------------------------------------------------\n");
    int networkSize = hiddenLayerNum + 1;
    Layer *network = (Layer *)calloc(networkSize, sizeof(Layer)); //Set network to num of layer
    network[0].inputVar = varColumn;                              //Number of Input Layer variables=number of variables in data
    network[networkSize - 1].numNeuron = 1;                       //networkSize-1=Output Layer, output layer will only have 1 neuron

    //Read in user input for network parameters
    for (int i = 0; i < hiddenLayerNum; i++) //Used hiddenLayerNum here to traverse as outputLayer has already been set
    {
        printf("Number of neurons in Hidden Layer [%d]: ", i + 1);
        scanf("%d", &network[i].numNeuron);
        network[i + 1].inputVar = network[i].numNeuron; //Num of inputs for current layer=Previous layer number of neurons
        printf("Number of Neurons in layer: %d", network[i].numNeuron);
        printf("\nNumber of inputs to each neuron: %d\n", network[i].inputVar);
        printf("----------------------------------------------------------------------------------------------------------\n");
    }

    //initializeNetwork(network, networkSize); //initialise all layers and neurons
    //printNetwork(network,networkSize);
    ////////////////////////////////////////////MACHINE LEARNING PARAMETERS SET UP END//////////////////////////////////////////////////////////////
    printf("ALL PARAMETERS SET!!");
    //////////////////////////////////////////////////////PROGRAM START////////////////////////////////////////////////////////////////////////////
    float MAE = 0, MMSE = 100;
    int iteration = 0;
    clock_t begin = clock(); //Program start time

    if (iteration == 0)
    {
        printf("\n----------------------------------------------------------------------------------------------------------");
        printf("\nSTART OF TRAINING PHASE");
        printf("\n----------------------------------------------------------------------------------------------------------\n");
        initializeNetwork(network, networkSize); //initialise all layers and neurons
        feedForward(network, networkSize, trainingVar, trainingOutput, trainingRow, varColumn);
        iteration++;
        MAE = calcMAE(trainingDes, trainingOutput, trainingRow); //test run, not part of final
        MMSE = calcMMSE(trainingDes, trainingOutput, trainingRow);

        while (MAE < MAETarget) //If data fails to "train" reinitialize first iteration
        {
            initializeNetwork(network, networkSize); //initialise all layers and neurons
            feedForward(network, networkSize, trainingVar, trainingOutput, trainingRow, varColumn);
            MAE = calcMAE(trainingDes, trainingOutput, trainingRow);
            MMSE = calcMMSE(trainingDes, trainingOutput, trainingRow);
        }
        printf("Iteration: %d               MAE: %f                 MMSE: %f", iteration, MAE, MMSE);
        printf("\n----------------------------------------------------------------------------------------------------------\n");
    }

    while (MAE > MAETarget)
    {
        iteration++;
        feedBack(network, networkSize, trainingDes, trainingOutput, trainingRow, varColumn, learningRate);
        feedForward(network, networkSize, trainingVar, trainingOutput, trainingRow, varColumn);
        MAE = calcMAE(trainingDes, trainingOutput, trainingRow);
        MMSE = calcMMSE(trainingDes, trainingOutput, trainingRow);
        printf("\nIteration: %d                       MAE:%f", iteration, MAE);
    }

    printf("\n----------------------------------------------------------------------------------------------------------");
    printf("\nTotal iterations: %d          MAE: %f          MMSE: %f", iteration, MAE, MMSE);
    printf("\n----------------------------------------------------------------------------------------------------------");
    printf("\nTRAINING COMPLETE");
    printf("\n----------------------------------------------------------------------------------------------------------");
    printf("\nSTART OF TESTING PHASE");
    printf("\n----------------------------------------------------------------------------------------------------------");
    feedForward(network, networkSize, testingVar, testingOutput, testingRow, varColumn);
    for (int i = 0; i < testingRow; i++)
    {
        if (testingOutput[i] == 0)
        {
            printf("\nTest Sample %d is normal", i + 1);
        }
        else
        {
            printf("\nTest Sample %d is abnormal", i + 1);
        } //*/
    }
    MAE = calcMAE(testingDes, testingOutput, testingRow);
    MMSE = calcMMSE(testingDes, testingOutput, testingRow);
    printf("\n----------------------------------------------------------------------------------------------------------");
    printf("\nTesting Output          MAE: %f          MMSE: %f", MAE, MMSE);
    printf("\n----------------------------------------------------------------------------------------------------------");
    printf("\nTraining Output Confusion Matrix");
    printf("\n----------------------------------------------------------------------------------------------------------");
    confusionMatrix(trainingDes, trainingOutput, trainingRow);
    printf("\n----------------------------------------------------------------------------------------------------------");
    printf("\nTesting Output Confusion Matrix");
    printf("\n----------------------------------------------------------------------------------------------------------");
    confusionMatrix(testingDes, testingOutput, testingRow);
    //////////////////////////////////////////////////////PROGRAM END////////////////////////////////////////////////////////////////////////////
    freeNetwork(network, networkSize);
    freeArrays(data, trainingVar, trainingDes, testingVar, testingDes, trainingOutput, testingOutput, dataRow, trainingRow, testingRow);
    fclose(infile_ptr);                                         //close file
    clock_t end = clock();                                      //Program end time
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; //calculate program run time
    printf("\nTime Taken: %.2lfs", time_spent);                 //print program run time
    printf("\n----------------------------------------------------------------------------------------------------------");
    return 0;
}

////////////////////////////////////////////////////////////FUNCTIONS////////////////////////////////////////////////////////////////////////////
void make2D(float **array, int row, int column)
{
    for (int i = 0; i < row; i++)
    {
        array[i] = (float *)calloc(column, sizeof(float));
    }
}

void free2D(float **array, int row)
{
    for (int i = 0; i < row; i++)
    {
        free(array[i]);
    }
    free(array);
}

void freeArrays(float **data, float **trainingVar, float *trainingDes, float **testingVar, float *testingDes, float *trainingOutput, float *testingOutput, int dataRow, int trainingRow, int testingRow)
{
    free2D(data, dataRow);
    free2D(trainingVar, trainingRow);
    free(trainingDes);
    free2D(testingVar, testingRow);
    free(testingDes);
    free(trainingOutput);
    free(testingOutput);
}

void initializeNetwork(Layer *network, int networkSize)
{
    for (int i = 0; i < networkSize; i++) //initialise weights and bias for all neurons
    {
        initializeLayer(&network[i]);
    }
}

void printNetwork(Layer *network, int networkSize)
{
    for (int i = 0; i < networkSize; i++)
    {
        for (int j = 0; j < network[i].numNeuron; j++)
        {
            for (int k = 0; k < network[i].inputVar; k++)
            {
                printf("\n%f", network[i].neuron[j].weight[k]);
            }
            printf("\n\nBias: %f\n", network[i].neuron[j].bias);
        }
        printf("\n--------------------");
    } //*/
}

void freeNetwork(Layer *network, int networkSize) //Free memory allocated to network
{
    for (int i = 0; i < networkSize; i++)
    {
        freeLayer(&network[i]);
    }
    free(network);
}