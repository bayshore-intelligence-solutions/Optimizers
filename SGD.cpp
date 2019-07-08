#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <float.h>
#include <math.h>

using namespace std;

double predict(vector<double>X, vector<double>pW, double pB){
    double sum = 0.0;
    for(int i=0;i<X.size();i++){
        sum += X[i]*pW[i];
    }
    return sum + pB;
}
bool min_step_size(vector<double> stepSizes){
    for(int i=0;i<stepSizes.size();i++){
        if(abs(stepSizes[i])>0.001)
            return false;
    }
    return true;
}

//vector A = vector A - vector B
vector<double> vector_subtraction(vector<double> a, vector<double> b){
    for(int i=0;i<a.size();i++){
        a[i] -= b[i];
    }
    return a;
}

//vector C = vector A x scaler B
vector<double> vector_scaler_multiplication(double value, vector<double> X){
    for(int i=0;i<X.size();i++){
        X[i] *= value;
    }
    return X;
}


void initalize(vector<double> &v) {
    srand (static_cast <unsigned> (time(0)));
    for (int i = 0; i <= v.size(); i++) {
        double r = static_cast <float> (rand()) / static_cast <double> (RAND_MAX);
        v[i] = r;
    }
}

int main(){

    vector<vector<double>> training_data;
    vector<vector<double>> validation_data;
    int row_count = 0;
    string line;
    ifstream dataset_file("ScaledBoston.csv");
    while(getline(dataset_file, line)){
        if(row_count == 0){
            row_count++;
            continue;
        }
        vector<double> row;
        stringstream ss(line);
        string value;
        int column_count = 0;
        while(getline(ss, value, ',')){
            if(column_count == 0){
                column_count++;
                continue;
            }
            double val = stod(value.c_str());
            row.push_back(val);
            column_count++;
        }
        if(row_count<351){
            training_data.push_back(row);
        }
        else{
            validation_data.push_back(row);
        }
        row_count++;
    }

    //initializing variables
    int max_iterations = 10000;
    double learning_rate = 0.01;
    vector<double> W(13, 0.0);
    initalize(W);
    double B = 0.0;
    vector<double> stepsize_W(13);
    vector<double> pW(13);
    vector<double> dW(13);
    double error = DBL_MAX;
    double pB;
    double dB;

    for(int i=0;i<=max_iterations;i++){
        //choose and store random data point
        int random_i = rand() % 350;
        // Getting one single point for SGD
        vector<double> X = training_data[random_i];
        // Store and Remove the class label
        double y = X.back();
        X.pop_back();
        
        cout<<"*********************** ITERATION "<<i+1<<" ***********************\n\n";

        //store the weights and bias
        pW = W;
        pB = B;

        //find the derivatives
        error = y - predict(X, pW, pB);
        // cout << error << endl;
        dW = vector_scaler_multiplication(-2*error, X);
        dB = -2*error;

        //calculate the step size for weights
        stepsize_W = vector_scaler_multiplication(learning_rate, dW);

        //update the weights and bias
        W = vector_subtraction(pW, stepsize_W);
        B = pB - (learning_rate*dB);

        //training error
        cout<<"Training error: "<<pow(y-predict(X, pW, pB), 2)<<"\n\n";

        if(min_step_size(stepsize_W)){
            //final weights and bias

            cout<<"Cannot converge more!" << endl;
            for(auto each: stepsize_W) {
                cout << each << endl;
            }
            cout << endl;
            break;
            //calculate validation error
        }
    }
    return 0;
}