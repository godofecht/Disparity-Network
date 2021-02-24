#include "model.h"

class disparityNet : public Model
{

    disparityNet()
    {
        vector<unsigned> topology;
        topology.push_back(10);
        topology.push_back(10);
        topology.push_back(1);

        SetTopology(topology);
        InitializeTopology();
    }

    
};