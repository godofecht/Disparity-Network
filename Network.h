#ifndef NETWORK_H
#define NETWORK_H

#include "NN.h"
#include <cmath>

class Network
{
	public:
		Network(const vector <unsigned> &topology);
	//	void backPropagate(const vector <double> &targetVals);
		void backPropagate(double m_error);
		void feedForward(vector <double> &inputVals);
		void getResults(vector <double> &resultVals);
		double getRecentAverageError(void) const { return m_recentAverageError; }

		vector<double> GetWeights() const;
		void PutWeights(vector<double> &weights);

	    void UpdateWeights();

		void NormalizeWeights(int connection_index);

		vector<Layer> GetLayers()
		{
 			return m_layers;
		}

		double V = 0.0;
		double U = 0.0;
		double F;
		double A = 1000.0;




    	vector<double> dudxs;
    	vector<double> dvdxs;


		vector<double> y_tilde;
		vector<double> y_bar;


		vector<double> Output_Array;


		double dudx = 0.0,dvdx = 0.0;


		vector<double> maskU;
		vector<double> maskV;
		vector<double> maskDashV;
		vector<double> maskDashU;

		vector<double> errors;


		double h_U = 32.0; //I need to move h out of the function parameters for getconvmaskdashU and getconvmaskdashV
		double h_V = 3200.0;




		void CalcV()
		{
			double sumDiffSquared = 0;
			for(int k=0;k<A;k++)
			{
				double diff = y_bar[k] - Output_Array[k];
				sumDiffSquared += pow(diff,2.0);
			}
			V = (sumDiffSquared)/2.0;
		}

		void CalcU()
		{
			double sumDiffSquared = 0.0;
			for(int k=0;k<A;k++)
			{
				double diff = y_tilde[k] - Output_Array[k];
				sumDiffSquared += pow(diff,2.0);
			}	
			U = (sumDiffSquared)/2.0;
		}


		void CalcF();


		double getDUDX(int a);

		double getDVDX(int a);



		double getErrorDerivative(double dvdx,double dudx); //technically merit derivative because dFdX = -dEdX


		vector<double> getConvolutionalMaskDashU(double h)
		{
			vector<double> kernel = getConvolutionalMaskU(h);
			int w = getW(h_U);
			kernel[0] = kernel[0] - 1.0;
			return kernel;
		}


		vector<double> getConvolutionalMaskDashV(double h)
		{
			vector<double> kernel = getConvolutionalMaskV(h);
			int w = getW(h_V);
			kernel[0] = kernel[0] - 1.0;
			return kernel;
		}

		vector<double> getConvolutionalMaskU(double h)
		{
			vector<double> kernelU;
			double kernelVal;
			int w = getW(h_U);

			kernelVal = 0.0;
			for (int x = -w; x <= w; x++)
			{
				kernelVal = exp(-getLambdaU(h_U) * (double)(x) * (double)(x));
				kernelU.push_back(kernelVal);



		//		cout<<x<<endl;
			
			}

			normalizeVector(kernelU);

			return kernelU;
		}

//fix masks
		vector<double> getConvolutionalMaskV(double h)
		{
			vector<double> kernelV;
			double kernelVal;
			int w = getW(h_V);
			kernelVal = 0.0;
			for (int x = -w; x <= w; x++)
			{
				//	cout<<(double)(getSub(a,x))<<endl;
				kernelVal = exp(-getLambdaV(h_V) * (double)(x) * (double)(x));
				//kernalVal += PRT


				kernelV.push_back(kernelVal);

			}

			normalizeVector(kernelV);
			return kernelV;
		}

		double getLambdaU(double h)
		{
			double l_u = log(2.0)/h_U;
			return l_u;
		}

		double getLambdaV(double h)
		{
			double l_v = log(2.0)/h_V;
			return l_v;
		}

		int getW(double h)
		{
			int w;
			if(4.0*h < (A/2.0-1.0))
			{
				w = 4.0*h;
			}
			else
			{
				w = A/2.0 - 1.0;
			}
			return w;
		}


	void normalizeVector(vector<double> &vec)
	{
		double sum = 0.0;
		for(int i=0;i<vec.size();i++)
		{
			sum += vec[i];
		}
		for(int i=0;i<vec.size();i++)
		{
			vec[i] /= sum;
		}
	}

	void CalcAverages()
	{
		for(int k=0;k<A;k++)
		{
			double sum_bar = 0.0;
			double sum_tilde = 0.0;
			for(int a=0;a<A;a++)
			{
				sum_bar += (maskV[(getSub(a,k))]*Output_Array[a]);
				sum_tilde += (maskU[(getSub(a,k))]*Output_Array[a]);	
			}
			y_bar[k] = sum_bar;
			y_tilde[k] = sum_tilde;
		}
	}



	int getSub(int a, int k)
	{
		int diff = a-k;
		if(diff <0.0)
			diff = 1000.0+diff;
		if(diff > 1000.0)
			diff = diff - 1000.0;
		return diff;
	}


		vector <Layer> m_layers;
	private:

		double m_gradient;
		double m_error;
		double m_recentAverageError;
		double m_recentAverageSmoothingFactor;
};

#endif