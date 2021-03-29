#ifndef NN_H
#define NN_H

#include<vector>
#include<cmath>
#include<cassert>
#include<iostream>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;
class connection
{
public:
	double weight;
	double deltaweight;



	void setDW(double dw)
	{
		deltaweight = dw;
	}
};
class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void feedForward(Layer &prevLayer);
	void feedForwardHidden(Layer &prevLayer);
	void feedForwardIO(Layer &prevLayer);
	double getOutputVal();
	void setOutputVal(double n);
	void calcHiddenGradients(const Layer &nextLayer);
	void calcOutputGradients(double targetVal);
	void updateInputWeights(Layer &prevLayer);
    double eta;
    double alpha;
	vector <connection> m_outputWeights;

	int getIndex();

    double UpdateYBar();
    double UpdateYTilde();

	// SFA stuff
	double y_bar = 0.0;
	double y_tilde = 0.0;
	
    double gamma_long = 0.0;
    double gamma_short = 0.0;

private:
	double m_gradient;
	double m_outputVal;
	static double randomWeight();
	unsigned m_myIndex;
	double sumDOW(const Layer &nextLayer) const;
	static double transferFunctionDerivative(double x);
	static double transferFunction(double x);	
};


#endif