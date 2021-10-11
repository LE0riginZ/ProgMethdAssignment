float calcSigmoid(float z)
{
    float sigmoid = 1 / (1 + expf(-z));
    return sigmoid;
}

void feedForward(Layer *network, int networkSize, float **data, float *output, int dataRow, int dataColumn)
{
    for (int i = 0; i < dataRow; i++)
    {
        //Input Layer
        for (int j = 0; j < network[0].numNeuron; j++)
        {
            float z = 0;
            float b = network[0].neuron[j].bias;
            for (int k = 0; k < network[0].inputVar; k++)
            {
                float w = network[0].neuron[j].weight[k];
                float x = data[i][k];
                network[0].neuron[j].input[k] = data[i][k]; //save input for feedback
                z += (x * w);
            }
            z += b;
            network[0].neuron[j].output = calcSigmoid(z);
        } //

        //Hidden Layers to Output Layer (Start from second layer)
        for (int j = 1; j < networkSize; j++)
        {

            for (int k = 0; k < network[j].numNeuron; k++)
            {
                float z = 0;
                float b = network[j].neuron[k].bias;
                for (int n = 0; n < network[j].inputVar; n++)
                {
                    float w = network[j].neuron[k].weight[n];
                    float x = network[j - 1].neuron[k].output;                           //Get output from previous layer neuron
                    network[j - 1].neuron[k].input[n] = network[j - 1].neuron[k].output; //save input for feedback
                    z += (x * w);
                }
                z += b;
                network[j].neuron[k].output = calcSigmoid(z);
            }
        }

        //Shift final output from machine learning to output array
        output[i] = roundf(network[networkSize - 1].neuron[0].output); //Take output from output layer neuron, neuron[0] as output layer only has 1 neuron
        //printf("\n%f", output[i]); //
    }
}

void feedBack(Layer *network, int networkSize, float *desired, float *output, int dataRow, int dataColumn, float learningRate)
{
    float errorOutput = 0;
    float errorHidden = 0;
    //Output layer weight change
    for (int i = 0; i < dataRow; i++)
    {
        float y = output[i];
        float b = network[networkSize - 1].neuron[0].bias;
        float d = desired[i];
        float z = 0;
        for (int j = 0; j < network[networkSize - 1].inputVar; j++)
        {
            float w = network[networkSize - 1].neuron[0].weight[j]; //output layer weight
            float x = network[networkSize - 1].neuron[0].input[j];  //output between output layer and hidden layer
            z += (x * w);                                           //summation of z values
        }
        z += b;
        errorOutput += (y - d) * (expf(z) / (powf(1 + expf(z), 2)));
    }
    errorOutput = errorOutput / 90;
    for (int i = 0; i < network[networkSize - 1].inputVar; i++)
    {
        network[networkSize - 1].neuron[0].weight[i] -= learningRate * errorOutput * network[networkSize - 1].neuron[0].weight[i];
    }
    network[networkSize - 1].neuron[0].bias -= learningRate * errorOutput;

    if (networkSize - 2 < 0) //If netowrk has 0 hidden layers, exit out of feedBack
    {
        return;
    }
    //Hidden layers weight change
    for (int i = networkSize - 2; i > -1; i--) //start from output layer-1, end at layer 0 (starting layer)
    {
        for (int n = 0; n < network[i + 1].numNeuron; n++) //for each neuron in the layer in front
        {
            for (int j = 0; j < network[i].numNeuron; j++) //for each neuron in current layer
            {
                //network[i + 1].neuron[n].weight[j] is the corresponding weight from the n neuron in front to current layer's neuron
                errorHidden = errorOutput * network[i + 1].neuron[n].weight[j] * (expf(network[i + 1].neuron[n].weight[j]) / (powf((1 + expf(network[i].neuron[j].output)), 2)));
                for (int k = 0; k < network[i].inputVar; k++) //for each weight in current neuron in current layer
                {
                    network[i].neuron[j].weight[k] -= learningRate * errorHidden * network[i].neuron[j].weight[k];
                }
                network[i].neuron[j].bias -= learningRate * errorHidden;
            }
        }
        errorOutput = errorHidden; //set current errorHidden as next layer's errorOutput
        errorHidden = 0;           //reset errorHidden for next layer
    }
}
