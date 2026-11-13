#include "model.h"
#include <fstream>
#include <cctype>
#include <sstream>
#include <istream>
#include <iostream>
#include <iterator>
#include <algorithm>
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
            // The rows are wrapped in double quotes; drop them so the first
            // value is not read as 0.
            s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
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

    double learningRate = 0.001;

    // Runs the network on all A windows, then works out U, V and F.
    double CalcObjective()
    {
        for (int a = 0; a < getNetwork()->A; a++)
        {
            inputVector = getInputVector(image1, image2, a);
            feedforward(inputVector);
            getNetwork()->Output_Array[a] = GetResult()[0];
        }
        getNetwork()->CalcAverages();
        getNetwork()->CalcU();
        getNetwork()->CalcV();
        getNetwork()->CalcF();
        return getNetwork()->F;
    }

    // Adds dF/dw for every weight to the weights' running totals.
    // Each window is fed forward again before backpropagation, so the
    // gradient for window a uses the neuron outputs for window a.
    void AccumulateGradient()
    {
        for (int a = 0; a < getNetwork()->A; a++)
        {
            getNetwork()->dudxs[a] = getNetwork()->getDUDX(a);
            getNetwork()->dvdxs[a] = getNetwork()->getDVDX(a);
            getNetwork()->errors[a] = getNetwork()->getErrorDerivative(getNetwork()->dvdxs[a], getNetwork()->dudxs[a]);
        }
        for (int a = 0; a < getNetwork()->A; a++)
        {
            inputVector = getInputVector(image1, image2, a);
            feedforward(inputVector);
            getNetwork()->backPropagate(getNetwork()->errors[a]);
        }
    }

    // One training round over all A windows. Returns F before the update.
    double Train(int current_epoch)
    {
        double F = CalcObjective();
        AccumulateGradient();
        getNetwork()->applyWeightChanges(learningRate);
        return F;
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
