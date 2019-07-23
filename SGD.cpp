#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <float.h>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>

using namespace std;

typedef long double ld;

// (X*W) + B
vector<ld> predict(vector<vector<ld>>X, vector<ld>pW, ld pB){
    int batch_size = X.size();
    int features = X[0].size();
    vector<ld> predicted(batch_size, 0.0);
    for(int i=0;i<batch_size;i++){
        for(int j=0;j<features;j++){
            predicted[i] += X[i][j]*pW[j];
        }
        predicted[i] += pB;
    }
    return predicted;
}

// for l1 regularization
vector<ld> predict_minus_j(vector<vector<ld>> X, vector<ld> W, ld B, int j){
    int batch_size = X.size();
    int features = X[0].size();
    vector<ld> predicted(batch_size, 0.0);
    for(int i=0;i<batch_size;i++){
        for(int k=0;k<features;k++){
            if(k == j){
                predicted[i] += 0;
            }
            else{
                predicted[i] += X[i][k]*W[k];
            }
        }
        predicted[i] += B;
    }
    return predicted;
}

// store mean of all features in a vector
vector<ld> mean_vector (vector<vector<ld>> dataset){
    int featureset_row = dataset.size();
    int featureset_col = dataset[0].size() - 1;
    vector<ld> mean(featureset_col, 0.0);
    for(int i=0;i<featureset_col;i++){
        ld sum = 0.0;
        for(int j=0;j<featureset_row;j++){
            sum += dataset[j][i];
        }
        mean[i] = sum/featureset_row;
    }
    return mean;
}

// store standard deviation of all features in a vector
vector<ld> standard_deviation_vector(vector<vector<ld>> dataset, vector<ld> mean){
    int featureset_row = dataset.size();
    int featureset_col = dataset[0].size() - 1;
    vector<ld> sd(featureset_col, 0.0);
    for(int i=0;i<featureset_col;i++){
        ld sum = 0.0;
        for(int j=0;j<featureset_row;j++){
            sum += pow((dataset[j][i]-mean[i]), 2);
        }
        sd[i] = pow((sum/featureset_row), 0.5);
    }
    return sd;
}

// vector A = vector A - vector B
vector<ld> vector_subtraction(vector<ld> a, vector<ld> b){
    for(int i=0;i<a.size();i++){
        a[i] -= b[i];
    }
    return a;
}

// vector A = vector A - vector B
vector<ld> vector_scaler_addition(ld a, vector<ld>b){
    for(int i=0;i<b.size();i++){
        b[i] += a;
    }
    return b;
}

// vector C = vector A x scaler B
vector<ld> vector_scaler_multiplication(ld value, vector<ld> X){
    for(int i=0;i<X.size();i++){
        X[i] *= value;
    }
    return X;
}

// transpose of a vector(matrix)
vector<vector<ld>> transpose_vector(vector<vector<ld>> X){
    int batch_size = X.size();
    int features = X[0].size();
    vector<vector<ld>> transpose(features, vector<ld>(batch_size));
    for(int i=0;i<batch_size;i++){
        for(int j=0;j<features;j++){
            transpose[j][i] = X[i][j];
        }
    }
    return transpose;
}

// vector multiplication, (N x M)*(M x 1) = N x 1
vector<ld> vector_multiplication(vector<ld> y, vector<vector<ld>> XT){
    int features = XT.size();
    int batch_size = XT[0].size();
    vector<ld> result(features, 0.0);
    for(int i=0;i<features;i++){
        for(int j=0;j<batch_size;j++){
            result[i] += XT[i][j]*y[j];
        }
    }
    return result;
}

// mean square error
ld MSE(vector<ld> y, vector<ld> predicted){
    int size = y.size();
    ld mse = 0.0;
    for(int i=0;i<size;i++){
        mse += pow((y[i] - predicted[i]), 2);
    }
    return mse/size;
}

// mean absolute error
ld MAE(vector<ld> y, vector<ld> predicted){
    int size = y.size();
    ld mae = 0.0;
    for(int i=0;i<size;i++){
        mae += abs(y[i] - predicted[i]);
    }
    return mae/size;
}

// Rsquared error
ld Rsquared(vector<ld> y, vector<ld> predicted){
    int size = y.size();
    ld sse = 0.0;
    ld sst = 0.0;
    ld y_mean = (accumulate(y.begin(), y.end(), (ld)0.0))/size;
    for(int i=0;i<size;i++){
        sse += pow((y[i] - predicted[i]), 2);
        sst += pow((y[i] - y_mean), 2);
    }
    return (1.0 - (sse/sst));
}

// initialize vector with random values
void initialize(vector<ld> &v) {
    srand (static_cast <unsigned> (time(nullptr)));
    for (int i = 0; i < v.size(); i++) {
        ld r = static_cast<float>(rand())/static_cast<ld>(RAND_MAX);
        v[i] = r;
    }
}

// scaling the dataset using Mean Normalization technique
vector<vector<ld>> scale(vector<vector<ld>> dataset, vector<ld> mean, vector<ld> sd){
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
void SGD(vector<vector<ld>> training_data, vector<vector<ld>> validation_data, const string& evaluation_metric = "mse", int batch_size = 100, const string& learning_rate = "adaptive", int max_iterations = 10000, ld eta0 = 0.1, bool early_stopping = true, int no_iter_change = 10, bool use_validation_error = true, ld tol = 0.001, bool logging = false, const string& regularization = "none", ld lambda = 0.5){
    // for logging to a file
    ofstream myfile;
    if(logging){
        myfile.open("/Users/bis/CLionProjects/test/output.log", ofstream::app);
        time_t now = time(nullptr);
        char* dt = ctime(&now);
        myfile<<"LOCAL DATE AND TIME: "<<dt<<"\n";
        myfile<<"Parameters:\n\tbatch_size = "<<batch_size<<"\n\tEvaluation_metric = "<<evaluation_metric<<"\n\tlearning_rate = "<<learning_rate<<"\n\tmax_iterations = "<<max_iterations<<"\n\teta0 = "<<eta0<<"\n\tearly_stopping = "<<early_stopping<<"\n\tno_iter_change = "<<no_iter_change<<"\n\tuse_validation_error = "<<use_validation_error<<"\n\ttolerence = "<<tol<<"\n\tregularization = "<<regularization<<"\n\tlambda = "<<lambda<<"\n\n";
    }

    // generating m = batch_size unique random numbers
    int n = training_data.size();
    int arr[n];
    for(int i=0;i<n;i++){
        arr[i] = i;
    }

    // initializing variables
    int features = training_data[0].size() - 1;

    // initialize X and y for validation set
    vector<vector<ld>> vX(validation_data.size(), vector<ld>(features));
    vector<ld> vy(validation_data.size());

    // if using validation error
    if(use_validation_error){
        for(int x=0;x<validation_data.size();x++){
            vector<ld> data = validation_data[x];
            vy[x] = data.back();
            data.pop_back();
            vX[x] = data;
        }
    }

    static vector<ld> W(features, 0.0);
    // initialize W with random values only for the first time
    if(accumulate(W.begin(), W.end(), (ld)0.0) == 0.0){
        initialize(W);
    }
    static ld B = 0.0;

    vector<ld> stepsize_W(features);
    vector<ld> pW(features);
    vector<ld> dW(features);
    vector<ld> dowW(batch_size, DBL_MAX);
    ld pB;
    ld dB;
    ld pE = INFINITY;
    ld error;

    for(int i=0;i<max_iterations;i++){
        // shuffle arr for random values
        auto rng = default_random_engine (random_device()());
        shuffle(arr, arr+n, rng);

        // initialize X and y
        vector<vector<ld>> X(batch_size, vector<ld>(features));
        vector<ld> y(batch_size);

        for(int x=0;x<batch_size;x++){
            // Getting random data point(s)
            vector<ld> data = training_data[arr[x]];
            // store and remove class label
            y[x] = data.back();
            data.pop_back();
            X[x] = data;
        }

        if(logging){
            myfile<<"*********************** ITERATION "<<i+1<<" ***********************\n\n";
        }
        cout<<"*********************** ITERATION "<<i+1<<" ***********************\n\n";

        // store the weights and bias
        pW = W;
        pB = B;

        // find the derivatives
        if(regularization == "none"){   // for no regularization
            dowW = vector_subtraction(y, predict(X, pW, pB));
            dW = vector_multiplication(dowW, transpose_vector(X));
            dW = vector_scaler_multiplication((-2.0/batch_size), dW);
            dB = (-2.0/batch_size)*accumulate(dowW.begin(), dowW.end(), (ld)0.0);

            // update the weights and bias
            W = vector_subtraction(pW, vector_scaler_multiplication(eta0, dW));
            B = pB - (eta0*dB);
        } else if(regularization == "l2"){  // for l2 regularization
            dowW = vector_subtraction(y, predict(X, pW, pB));
            dW = vector_multiplication(dowW, transpose_vector(X));
            dW = vector_scaler_multiplication((-2.0/batch_size), dW);
            ld penalty_weight = 2.0*lambda*accumulate(pW.begin(), pW.end(), (ld)0.0);
            dW = vector_scaler_addition(penalty_weight, dW);
            dB = (-2.0/batch_size)*accumulate(dowW.begin(), dowW.end(), (ld)0.0);

            // update the weights and bias
            W = vector_subtraction(pW, vector_scaler_multiplication(eta0, dW));
            B = pB - (eta0*dB);
        } else if(regularization == "l1"){  // for l1 regularization
            for(int w=0;w<W.size();w++){
                vector<ld> predicted = predict_minus_j(X, W, pB, w);
                ld diff = ((accumulate(y.begin(), y.end(), (ld)0.0)/batch_size) - (accumulate(predicted.begin(), predicted.end(), (ld)0.0)/batch_size));
                if(diff < (-lambda/2)){
                    W[w] = diff + (lambda/2);
                } else if(diff > (lambda/2)){
                    W[w] = diff - (lambda/2);
                } else {
                    W[w] = 0;
                }
            }
            dowW = vector_subtraction(y, predict(X, pW, pB));
            dB = (-2.0/batch_size)*accumulate(dowW.begin(), dowW.end(), (ld)0.0);
            B = pB - (eta0*dB);
        }

        // invscaling (inverse scaling) learning rate
        if(learning_rate == "invscaling"){
            eta0 /= pow(i+1, 0.5);
        }

        // training error
        ld training_error = 0;
        if(evaluation_metric == "mae"){
            training_error = MAE(y, predict(X, W, B));
        }
        else if(evaluation_metric == "rsquared"){
            training_error = Rsquared(y, predict(X, W, B));
        }
        else {
            training_error = MSE(y, predict(X, W, B));
        }

        // validation error
        ld validation_error = 0;
        if(use_validation_error){
            if(evaluation_metric == "mae"){
                validation_error = MAE(vy, predict(vX, W, B));
            }
            else if(evaluation_metric == "rsquared"){
                validation_error = Rsquared(vy, predict(vX, W, B));
            }
            else {
                validation_error = MSE(vy, predict(vX, W, B));
            }
        }

        if(logging){
            myfile<<"Training error: "<<training_error<<"\n";
            if(use_validation_error){
                myfile<<"Validation error: "<<validation_error<<"\n\n";
            }
        }
        cout<<"Training error: "<<training_error<<"\n\n";
        if(use_validation_error){
            cout<<"Validation error: "<<validation_error<<"\n\n";
            error = validation_error;
        } else {
            error = training_error;
        }

        // for Rsquared evaluation metric
        if(evaluation_metric == "rsquared"){
            error = -error;
        }

        if(early_stopping){
            if(error >= pE-tol && no_iter_change == 0){
                if(logging){
                    myfile<<"Cannot converge more!\n";
                    myfile<<"Training error: "<<training_error;
                    if(use_validation_error){
                        myfile<<"\nValidation error: "<<validation_error<<"\n";
                    }
                }
                cout<<"Cannot converge more!\n";
                cout<<"Training error: "<<training_error;
                if(use_validation_error){
                    cout<<"\nValidation error: "<<validation_error<<"\n\n";
                }
                break;
            }
            else if(error >= pE-tol && no_iter_change != 0){
                no_iter_change--;
                if(learning_rate == "adaptive"){
                    eta0 /= 5;
                }
            }
        }
        else if(learning_rate == "adaptive" && error >= pE-tol){
            eta0 /= 5;
        }

        pE = error;
    }

    myfile<<"\n\n~~~~End of Output~~~~\n\n\n";
    myfile.close();
}

// preprocess the data and call SGD
void preprocess_and_call_SGD(const vector<vector<ld>>& dataset){
    vector<vector<ld>> training_data;
    vector<vector<ld>> validation_data;
    vector<ld> mean = mean_vector(dataset);
    vector<ld> sd = standard_deviation_vector(dataset, mean);
    vector<vector<ld>> scaled_dataset = scale(dataset, mean, sd);

    int n = scaled_dataset.size();
    int arr[n];
    for(int i=0;i<n;i++){
        arr[i] = i;
    }
    auto rng = default_random_engine (random_device()());
    shuffle(arr, arr+n, rng);

    int size = int(0.7*n);

    for(int i=0;i<n;i++){
        if(i<size){
            training_data.push_back(scaled_dataset[arr[i]]);
        }
        else{
            validation_data.push_back(scaled_dataset[arr[i]]);
        }
    }

    SGD(training_data, validation_data, "mse", 1, "adaptive", 1000, 1.0, true, 5, true, 0.001, false, "none", 0.1);
}

// read data from csv file into vector
void read_csv_data(bool buffering = false){
    //read data from csv file
    vector<vector<ld>> dataset;
    int row_count = 0;
    string line;
    bool ignore_first_column = false;
    ifstream dataset_file("/Users/bis/CLionProjects/test/satgpa.csv");

    // declare a buffer
    int buffer_size = 1;

    while(getline(dataset_file, line)){
        vector<ld> row;
        stringstream ss(line);
        string value;
        int column_count = 0;
        if(row_count == 0){
            getline(ss, value, ',');
            if(value == "\"\"" || value.empty() || value.length() == 0){
                ignore_first_column = true;
            }
            row_count++;
            continue;
        }
        while(getline(ss, value, ',')){
            if(!column_count && ignore_first_column){
                column_count++;
                continue;
            }
            //convert string values into ld values
            ld val = stold(value);
            row.push_back(val);
            column_count++;
        }
        dataset.push_back(row);

        if(buffering){
            // read the file in chunks of buffer size 1000
            if(buffer_size == 1000){
                preprocess_and_call_SGD(dataset);

                // set buffer_size to 0 after the buffer is full
                buffer_size = 0;

                // clear the dataset for new batch
                dataset.clear();
            }
            buffer_size++;
        }
    }

    if(!buffering){
        preprocess_and_call_SGD(dataset);
    }
}



// dummy functions below!

// read data from csv file into vector
void read_csv_data_test(bool buffering = false){
    //read data from csv file
    vector<vector<string>> dataset;
    int row_count = 0;
    string line;
    bool ignore_first_column = false;
    ifstream dataset_file("/Users/bis/Downloads/2017_Yellow_Taxi_Trip_Data.csv");

    // declare a buffer
    int buffer_size = 1;

    int batch_number = 0;

    while(getline(dataset_file, line)){
        vector<string> row;
        stringstream ss(line);
        string value;
        int column_count = 0;
        if(row_count == 0){
            getline(ss, value, ',');
            if(value == "\"\"" || value.empty() || value.length() == 0){
                ignore_first_column = true;
            }
            row_count++;
            continue;
        }
        while(getline(ss, value, ',')){
            if(!column_count && ignore_first_column){
                column_count++;
                continue;
            }
            //convert string values into ld values
            row.push_back(value);
            column_count++;
        }
        dataset.push_back(row);

        if(buffering){
            // read the file in chunks of buffer size 1000
            if(buffer_size == 1000){
                cout<<"Processing batch number "<<++batch_number<<" of "<<dataset.size()<<" rows\n";

                // set buffer_size to 0 after the buffer is full
                buffer_size = 0;

                // clear the dataset for new batch
                dataset.clear();

            }
            buffer_size++;
        }
    }

    cout<<"(last)Processing batch number "<<++batch_number<<" of "<<dataset.size()<<" rows\n";
}

int main(){
    read_csv_data(false);
    return 0;
}