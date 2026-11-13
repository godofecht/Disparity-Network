// Checks the backpropagated gradient dF/dw against a finite difference.
// Build: g++ -std=c++17 gradient_check.cpp Network.cpp NN.cpp -o gradientCheck
#include "disparityNet.h"
#include <cmath>

int main()
{
    DisparityNet dNet;
    Network *net = dNet.getNetwork();

    dNet.CalcObjective();
    dNet.AccumulateGradient();
    vector<double> analytic = net->GetWeightChanges();
    net->applyWeightChanges(0.0); // clears the totals, leaves the weights alone

    vector<double> weights = net->GetWeights();
    const double eps = 1e-5;
    double worst = 0.0;
    int checked = 0;
    // Input-to-hidden weights are 0..99; hidden-to-output weights are 100, 110, ..., 190.
    for (int i = 0; i < 200; i += 7)
    {
        if (i >= 100 && i % 10 != 0)
            continue;
        vector<double> w = weights;
        w[i] = weights[i] + eps;
        net->PutWeights(w);
        double fPlus = dNet.CalcObjective();
        w[i] = weights[i] - eps;
        net->PutWeights(w);
        double fMinus = dNet.CalcObjective();
        double numeric = (fPlus - fMinus) / (2.0 * eps);
        double relErr = fabs(numeric - analytic[i]) / fmax(1e-8, fabs(numeric) + fabs(analytic[i]));
        worst = fmax(worst, relErr);
        checked++;
        cout << "weight " << i << "  backprop " << analytic[i] << "  finite difference " << numeric
             << "  relative error " << relErr << endl;
    }
    net->PutWeights(weights);
    cout << checked << " weights checked, worst relative error " << worst << endl;
    return worst < 1e-4 ? 0 : 1;
}
