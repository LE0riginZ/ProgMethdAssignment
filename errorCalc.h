float calcMMSE(float *desired, float *output, int numRow)
{
    float summation = 0;
    float MMSE = 0;
    for (int i = 0; i < numRow; i++)
    {
        summation += powf(output[i] - desired[i], 2);
    }
    MMSE = summation / numRow;
    //printf("\nMMSE: %f", MMSE);

    return MMSE;
}
float calcMAE(float *desired, float *output, int numRow)
{
    float summation = 0;
    float MAE = 0;
    for (int i = 0; i < numRow; i++)
    {
        summation += fabs(output[i] - desired[i]);
    }
    MAE = summation / numRow;
    //printf("MAE: %f", MAE);
    return MAE;
} //*/

void confusionMatrix(float *desired, float *output, int numRow){
    float truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0, accuracy = 0;
    for (int i = 0; i < numRow; i++)
    {
        if (desired[i] == 1 && desired[i] == output[i])
        {
            truePositive++;
        }
        else if (desired[i] == 0 && desired[i] == output[i])
        {
            trueNegative++;
        }
        else if (desired[i] == 1 && desired[i] != output[i])
        {
            falseNegative++;
        }
        else if (desired[i] == 0 && desired[i] != output[i])
        {
            falsePositive++;
        }
    } //*/
    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative) * 100;
    printf("\nTrue Positive: %.0f           False Positive: %.0f", truePositive, falsePositive);
    printf("\nFalse Negative: %.0f          True Negative: %.0f", falseNegative, trueNegative);
    printf("\n----------------------------------------------------------------------------------------------------------");
    printf("\nAccuracy: %.2f%%", accuracy);
    printf("\n----------------------------------------------------------------------------------------------------------");
}