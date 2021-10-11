typedef struct neuron
{
    float bias;
    float output;  //save neuron output
    float *weight; //neuron weights
    float *input;  //save inputs that go into neuron
} Neuron;

typedef struct layer
{
    int inputVar;   //num of inputs into layer
    int numNeuron;  //num of neurons in layer
    Neuron *neuron; //array of neurons in layer
} Layer;

Neuron initializeNeuron(Neuron *neuron, int inputNum)
{
    neuron->weight = (float *)calloc(inputNum, sizeof(float)); //dynamically allocate array size based on user input
    if (neuron->weight == NULL)
    {
        fprintf(stderr, "Memory Allocation Failed");
        exit(0);
    }
    neuron->input = (float *)calloc(inputNum, sizeof(float)); //dynamically allocate array to save input data
    if (neuron->input == NULL)
    {
        fprintf(stderr, "Memory Allocation Failed");
        exit(0);
    }

    for (int i = 0; i < inputNum; i++)
    {
        neuron->weight[i] = rand() % 3 - 1; //fill weight member of neuron with random values from -1<=x<=1;
    }

    neuron->bias = rand() % 3 - 1; //fill bias member of neuron with random values from -1<=x<=1;
}

void initializeLayer(Layer *layer)
{
    layer->neuron = (Neuron *)calloc(layer->numNeuron, sizeof(Neuron));
    if (layer->neuron == NULL)
    {
        fprintf(stderr, "Memory Allocation Failed");
        exit(0);
    }
    for (int i = 0; i < layer->numNeuron; i++)
    {
        initializeNeuron(&layer->neuron[i], layer->inputVar);
    }
}

void freeLayer(Layer *layer) //free all memory used by neuron header
{
    for (int i = 0; i < layer->numNeuron; i++)
    {
        free(layer->neuron[i].weight);
    }
    free(layer->neuron);
}
