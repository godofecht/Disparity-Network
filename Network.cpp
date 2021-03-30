#include "Network.h"
#include "NN.cpp"
#include <vector>

Network::Network(const vector <unsigned> &topology)
{
	srand((unsigned int)time(NULL));

	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		m_layers.push_back(Layer());
		unsigned numOutputs = topology[layerNum];// layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have a new layer, now fill it with neurons
		for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; neuronNum++) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
		}
	}

	for(int i=0;i<A;i++)
	{
		Output_Array.push_back(0.0);
		y_tilde.push_back(0.0);
		y_bar.push_back(0.0);
		errors.push_back(0.0);
		dudxs.push_back(0.0);
		dvdxs.push_back(0.0);
	}

	maskU = getConvolutionalMaskU(h_U);
	maskV = getConvolutionalMaskV(h_V);

	maskDashU = getConvolutionalMaskDashU(h_U);
	maskDashV = getConvolutionalMaskDashV(h_V);
}

double Network::getErrorDerivative(double dvdx,double dudx)
{
	double v1 = 1.0/V;
	double u1 = 1.0/U;

	long deriv = ((v1) * dvdx) - ((u1) * dudx);
	
//	cout<<deriv<<endl;
	return deriv;
}

void Network::CalcF()
{
	F = log(V/U);
	cout<<U<<endl;
}

void Network::backPropagate(double m_error)
{
	Layer &outputLayer = m_layers.back();
	// Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size(); n++) {
		outputLayer[n].calcOutputGradients(m_error);
	}
	// Calculate hidden layer gradients
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) 
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];
		for (unsigned n = 0; n < hiddenLayer.size(); n++) 
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	// For all layers from outputs to first hidden layer,
	//5. Calculate weight changes for all weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--) 
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < layer.size(); n++) 
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

vector<double>  Network::GetWeights() const
{
	//this will hold the weights
	vector<double> weights;
	//for each layer
	for (int i = 0; i<m_layers.size(); i++)
	{
		//for each neuron
		for (int j = 0; j<m_layers[i].size(); j++)
		{
			//for each weight
			for (int k = 0; k<m_layers[i][j].m_outputWeights.size(); k++)
			{
				weights.push_back(m_layers[i][j].m_outputWeights[k].weight);
			}
		}
	}
	return weights;
}

//I think k has to do with overall num of neurons but not sure

double Network::getDVDX(int a) //done, maybe optimize later
{
	long sumDiff = 0.0;
	for(int k=0;k<A;k++)
	{
		double diff = y_bar[k] - Output_Array[k];
		
		if(a == k)
			sumDiff += diff * maskV[(getSub(a,k))] - 1.0;
		else
			sumDiff += diff * maskV[(getSub(a,k))];
	}
	return sumDiff;
}

double Network::getDUDX(int a) // done, maybe optimize later
{
	long sumDiff = 0.0;
	for(int k=0;k<A;k++)
	{
		double diff = y_tilde[k] - Output_Array[k];

		if(a == k)
			sumDiff += diff * maskU[(getSub(a,k))] - 1.0;
		else
			sumDiff += diff * maskU[(getSub(a,k))];
	//	cout<<maskDashU[(getSub(a,k))]<<endl;

	}
	return sumDiff;
}

//Double checked functions
//This is done
void Network::feedForward(vector <double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size());
	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); i++) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//Feed forward from input layer to hidden layer
	for (unsigned n = 0; n < m_layers[1].size(); n++) 
	{
		m_layers[1][n].feedForwardHidden(m_layers[0]);
	}

	//feed forward from hidden layer to output layer
	for (unsigned n = 0; n < m_layers[2].size(); n++) 
	{
		m_layers[2][n].feedForwardIO(m_layers[1]);
	}
}


void  Network::getResults(vector <double> &resultVals)
{
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size(); n++) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Network::PutWeights(vector<double> &weights)
{
	int cWeight = 0.0;
	//for each layer
	for (int i = 0; i<m_layers.size(); i++)
	{
		//for each neuron
		for (int j = 0; j<m_layers[i].size(); j++)
		{
			//for each weight
			for (int k = 0; k<m_layers[i][j].m_outputWeights.size(); k++)
			{
				m_layers[i][j].m_outputWeights[k].weight = weights[cWeight++];
			}
		}
	}
}