#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <float.h>
#include <math.h>
#include <numeric>
#include <random>

using namespace std;

vector<double> predict(vector<vector<double>>X, vector<double>pW, double pB){
    int batch_size = X.size();
    int features = X[0].size();
    vector<double> predicted(batch_size, 0.0);
    for(int i=0;i<batch_size;i++){
        for(int j=0;j<features;j++){
            predicted[i] += X[i][j]*pW[j];
        }
        predicted[i] += pB;
    }
    return predicted;
}

bool min_step_size(vector<double> stepSizes){
    for(int i=0;i<stepSizes.size();i++){
        if(abs(stepSizes[i])>0.001)
            return false;
    }
    return true;
}

//store mean of all features in a vector
vector<double> mean_vector (vector<vector<double>> dataset){
    int featureset_row = dataset.size();
    int featureset_col = dataset[0].size() - 1;
    vector<double> mean(featureset_col, 0.0);
    for(int i=0;i<featureset_col;i++){
        double sum = 0.0;
        for(int j=0;j<featureset_row;j++){
            sum += dataset[j][i];
        }
        mean[i] = sum/featureset_row;
    }
    return mean;
}

//store standard deviation of all features in a vector
vector<double> standard_deviation_vector(vector<vector<double>> dataset, vector<double> mean){
    int featureset_row = dataset.size();
    int featureset_col = dataset[0].size() - 1;
    vector<double> sd(featureset_col, 0.0);
    for(int i=0;i<featureset_col;i++){
        double sum = 0.0;
        for(int j=0;j<featureset_row;j++){
            sum += pow((dataset[j][i]-mean[i]), 2);
        }
        sd[i] = pow((sum/featureset_row), 0.5);
    }
    return sd;
}

// vector A = vector A - vector B
vector<double> vector_subtraction(vector<double> a, vector<double> b){
    for(int i=0;i<a.size();i++){
        a[i] -= b[i];
    }
    return a;
}

// vector C = vector A x scaler B
vector<double> vector_scaler_multiplication(double value, vector<double> X){
    for(int i=0;i<X.size();i++){
        X[i] *= value;
    }
    return X;
}

// transpose of a vector(matrix)
vector<vector<double>> transpose_vector(vector<vector<double>> X){
    int batch_size = X.size();
    int features = X[0].size();
    vector<vector<double>> transpose(features, vector<double>(batch_size));
    for(int i=0;i<batch_size;i++){
        for(int j=0;j<features;j++){
            transpose[j][i] = X[i][j];
        }
    }
    return transpose;
}

// vector multiplication, (N x M)*(M x 1) = N x 1 
vector<double> vector_multiplication(vector<double> y, vector<vector<double>> XT){
    int features = XT.size();
    int batch_size = XT[0].size();
    vector<double> result(features, 0.0);
    for(int i=0;i<features;i++){
        for(int j=0;j<batch_size;j++){
            result[i] += XT[i][j]*y[j];
        }
    }
    return result;
}

// mean square error
double MSE(vector<double> y, vector<double> predicted){
    int size = y.size();
    double mse = 0.0;
    for(int i=0;i<size;i++){
        mse += pow((y[i] - predicted[i]), 2);
    }
    return mse/size;
}

// initialize vector with random values
void initalize(vector<double> &v) {
    srand (static_cast <unsigned> (time(0)));
    for (int i = 0; i <= v.size(); i++) {
        double r = static_cast<float>(rand())/static_cast<double>(RAND_MAX);
        v[i] = r;
    }
}

// scaling the dataset using Mean Normalization technique
vector<vector<double>> scale(vector<vector<double>> dataset, vector<double> mean, vector<double> sd){
    int featureset_row = dataset.size();
    int featureset_col = dataset[0].size() - 1;
    for(int i=0;i<featureset_col;i++){
        for(int j=0;j<featureset_row;j++){
            dataset[j][i] = (dataset[j][i] - mean[i])/sd[i];
        }
    }
    return dataset;
}

// SGD
void SGD(vector<vector<double>> training_data, int batch_size = 10, int max_iterations = 1000, double learning_rate = 0.01){
    // generating m = batch_size unique random numbers
    int n = training_data.size();
    int arr[n];
    for(int i=0;i<n;i++){
        arr[i] = i;
    }

    // initializing variables
    int features = training_data[0].size() - 1;
    
    vector<double> W(features, 0.0);
    // initialize W with random values
    initalize(W);
    double B = 0.0;

    vector<double> stepsize_W(features);
    vector<double> pW(features);
    vector<double> dW(features);
    vector<double> error(batch_size, DBL_MAX);
    double pB;
    double dB;

    for(int i=0;i<max_iterations;i++){
        // shuffle arr for random values
        auto rng = default_random_engine {};
        shuffle(arr, arr+n, rng);

        // initialize X and y
        vector<vector<double>> X(batch_size, vector<double>(features));
        vector<double> y(batch_size);

        for(int x=0;x<batch_size;x++){
            // Getting random data points for batch-SGD
            vector<double> data = training_data[arr[x]];
            // store and remove class label
            y[x] = data.back();
            data.pop_back();
            X[x] = data;
        }
        
        cout<<"*********************** ITERATION "<<i+1<<" ***********************\n\n";

        //store the weights and bias
        pW = W;
        pB = B;

        //find the derivatives
        error = vector_subtraction(y, predict(X, pW, pB));
        dW = vector_multiplication(error, transpose_vector(X));
        dW = vector_scaler_multiplication((-2.0/batch_size), dW);
        dB = (-2.0/batch_size)*accumulate(error.begin(), error.end(), 0.0);

        //calculate the step size for weights
        stepsize_W = vector_scaler_multiplication(learning_rate, dW);

        //update the weights and bias
        W = vector_subtraction(pW, stepsize_W);
        B = pB - (learning_rate*dB);

        // training error
        cout<<"Training error: "<<MSE(y, predict(X, pW, pB))<<"\n\n";

        if(min_step_size(stepsize_W)){
            //final weights and bias
            cout<<"Cannot converge more!" << endl;
            for(auto i: W){
                cout<<i<<" ";
            }
            cout<< B << "\n\n" << endl;
            break;
        }
    }
}

int main(){
    //read data from csv file
    vector<vector<double>> dataset;
    vector<vector<double>> training_data;
    vector<vector<double>> validation_data;
    int row_count = 0;
    string line;
    ifstream dataset_file("Boston.csv");
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
            //convert string values into double values
            double val = stod(value.c_str());
            row.push_back(val);
            column_count++;
        }
        dataset.push_back(row);
    }

    vector<double> mean = mean_vector(dataset);
    vector<double> sd = standard_deviation_vector(dataset, mean);
    vector<vector<double>> scaled_dataset = scale(dataset, mean, sd);
    
    int size = 0.7*scaled_dataset.size();

    for(int i=0;i<scaled_dataset.size();i++){
        if(i<size){
            training_data.push_back(scaled_dataset[i]);
        }
        else{
            validation_data.push_back(scaled_dataset[i]);
        }
    }

    SGD(training_data, 1);
    return 0;
}