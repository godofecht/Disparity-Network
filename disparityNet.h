#include "model.h"
#include <fstream>
#include <cctype>
#include <sstream>
#include <istream>
#include <iostream>
#include <iterator>
//#include <algorithm>
using std::istringstream;

class DisparityNet : public Model
{
public:
    vector<double> inputVector;

    vector<vector<double>> image1, image2;

    int dispCounter = 0;
    bool dispIncrease = true;

    int virtual_index = 0;

    vector<double> del_weights_ij, del_weights_jk;

    DisparityNet()
    {

        image1 = readCSV("image1.csv");
        image2 = readCSV("image2.csv");
        vector<unsigned> topology;
        topology.push_back(10);
        topology.push_back(10);
        topology.push_back(1);

        SetTopology(topology);
        InitializeTopology();

        vector<double> weights = getNetwork()->GetWeights();
        getNetwork()->normalizeVector(weights);
        getNetwork()->PutWeights(weights);

        for (int i = 0; i < getNetwork()->A; i++)
        {
            del_weights_ij.push_back(0.0f);
            del_weights_jk.push_back(0.0f);
        }
    }

    /*
* Parses through csv file line by line and returns the data
* in vector of vector of strings.
*/

    vector<double> getInputVector(std::vector<std::vector<double>> img1, std::vector<std::vector<double>> img2, int time_step)
    {
        vector<double> iVector;
        int start_index = (5 + 2) * (time_step);
        for (int i = 0; i < 5; i++)
        {
            iVector.push_back(img1[0][start_index + i]); //first element of each row
        }
        for (int i = 0; i < 5; i++)
        {
            iVector.push_back(img2[0][start_index + i]); //first element of each row
        }

        return iVector;
    }

    vector<std::string> split(std::string text, char delim)
    {
        std::string line;
        std::vector<std::string> vec;
        std::stringstream ss(text);
        while (std::getline(ss, line, delim))
        {
            vec.push_back(line);
        }
        ss.clear();
        return vec;
    }

    vector<std::vector<double>> readCSV(string fileName)
    {
        ifstream file(fileName);
        vector<std::vector<double>> dataList;
        string s = "";
        // Iterate through each line and split the content using delimeter

        while (getline(file, s))
        {
            vector<double> doubleArray;
            vector<std::string> vstrings = split(s, ' ');

            for (int i = 0; i < vstrings.size(); i++)
            {
                string a = vstrings[i];
                doubleArray.push_back(atof(a.c_str())); //stod doesn't work. This is the best solution possible.
                //update stackoverflow on solution above.
            }
            dataList.push_back(doubleArray);
        }

        // Close the File
        file.close();
        return dataList;
    }

    void Train(int current_epoch) //in each epoch there are 1000 time steps
    {
        //1. Calculate Output of all 1000 Virtual Units

        //      cout<<"STARTING NEW EPOCH"<<endl;
        for (int a = 0; a < getNetwork()->A; a++)
        {
            inputVector.clear();
            inputVector = getInputVector(image1, image2, a);
            feedforward(inputVector); //feeds inputs through , propagates and sets neuronal output values for calculation of V and U
            getNetwork()->Output_Array[a] = GetResult()[0];
        }
        getNetwork()->CalcAverages();

        //2. Calculate U and V
        getNetwork()->CalcU();
        getNetwork()->CalcV();
        getNetwork()->CalcF();

        double error = 0.0;

        double sum_del = 0.0;
        for (int a = 0; a < getNetwork()->A; a++)
        {
            getNetwork()->dudxs[a] = getNetwork()->getDUDX(a);
            getNetwork()->dvdxs[a] = getNetwork()->getDVDX(a);

            //3. Calculate errors for all output units
            getNetwork()->errors[a] = getNetwork()->getErrorDerivative(getNetwork()->dvdxs[a], getNetwork()->dudxs[a]);
            error += getNetwork()->errors[a];
 //           del_weights_jk[a] = getNetwork()->errors[a] * getNetwork()->Output_Array[a];
 //           sum_del += del_weights_jk[a];
 //cout<<getNetwork()->errors[a]<<endl;

        //    cout<<getNetwork()->dudxs[a]<<endl;
        }
        getNetwork()->backPropagate(error);
    }

    void write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<double>>> dataset)
    {
        // Make a CSV file with one or more columns of integer values
        // Each column of data is represented by the pair <column name, column data>
        //   as std::pair<std::string, std::vector<int>>
        // The dataset is represented as a vector of these columns
        // Note that all columns should be the same size

        // Create an output filestream object
        std::ofstream myFile(filename);

        // Send column names to the stream
        for (int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).first;
            if (j != dataset.size() - 1)
                myFile << ","; // No comma at end of line
        }
        myFile << "\n";

        // Send data to the stream
        for (int i = 0; i < dataset.at(0).second.size(); ++i)
        {
            for (int j = 0; j < dataset.size(); ++j)
            {
                myFile << dataset.at(j).second.at(i);
                if (j != dataset.size() - 1)
                    myFile << ","; // No comma at end of line
            }
            myFile << "\n";
        }

        // Close the file
        myFile.close();
    }
};

//need to init bias neuron in neuron.cpp
