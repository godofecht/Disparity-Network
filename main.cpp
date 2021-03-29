
#include "disparityNet.h"

using namespace std;


int NUM_EPOCHS = 100;

int main()
{
    DisparityNet dNet;

    vector<double> outputs;

    for(int i=0;i<NUM_EPOCHS;i++)
    {
        dNet.Train(i);
    }


    std::vector<std::pair<std::string,
    std::vector<double>>> vals = {{"Values", dNet.getNetwork()->Output_Array}};
    dNet.write_csv("outputs.csv", vals);


    return 0;
}





//correct for aspect ratio
//