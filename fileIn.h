int countRow(FILE *input)
{
    char c;
    int count = 0;
    for (c = getc(input); c != EOF; c = getc(input))
    {
        if (c == '\n') // Increment count if this character is new line
            count += 1;
    }
    fseek(input, 0, SEEK_SET); //reset pointer to start of file
    return count;
}

int countColumn(FILE *input)
{
    char c;
    int count = 1;                                    //starts from 1 as first variable does not have a comma
    for (c = getc(input); c != '\n'; c = getc(input)) //counts number of commas in the first line of the file
    {
        if (c == ',') // Increment count if this character is a comma
            count += 1;
    }
    fseek(input, 0, SEEK_SET); //reset pointer to start of file
    return count;
}

void populateData(FILE *input, float **data, int row, int column)
{
    //read file data into 2D array
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            fscanf(input, "%f,", &data[i][j]);
        }
    }
}

void printData(float **data, int row, int column)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            printf("%.2f ", data[i][j]);
        }
        printf("\n"); //print results to check
    }
}

void splitData(float **data, float **trainingVar, float *trainingDes, float **testingVar, float *testingDes, int dataRow, int dataColumn, int trainingRow, int varColumn)
{
    //filling up training sets
    for (int i = 0; i < trainingRow; i++)
    {
       for (int j=0;j<varColumn;j++){
           trainingVar[i][j]=data[i][j];
       }
       trainingDes[i]=data[i][dataColumn-1];
    }

    //filling up testing sets
    for (int i = trainingRow; i < dataRow; i++)
    {
       for (int j=0;j<varColumn;j++){
           testingVar[i-trainingRow][j]=data[i][j];
       }
       testingDes[i-trainingRow]=data[i][dataColumn-1];
    }
}