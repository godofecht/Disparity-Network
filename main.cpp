
#include "disparityNet.h"

using namespace std;


int NUM_EPOCHS = 100;

int main()
{
    DisparityNet dNet;

    vector<double> outputs;

    // Each line is the round number and F = log(V/U). Round 0 is before training.
    cout << 0 << " " << dNet.CalcObjective() << endl;
    for(int i=0;i<NUM_EPOCHS;i++)
    {
        double F = dNet.Train(i);
        cout << i + 1 << " " << F << endl;
    }

    // Recompute the outputs with the final weights before saving them.
    dNet.CalcObjective();

    std::vector<std::pair<std::string,
    std::vector<double>>> vals = {{"Values", dNet.getNetwork()->Output_Array}};
    dNet.write_csv("outputs.csv", vals);


    return 0;
}





//correct for aspect ratio
//