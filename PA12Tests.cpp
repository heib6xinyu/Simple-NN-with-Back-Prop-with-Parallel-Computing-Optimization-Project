#include "./data/DataSet.h"
#include "./network/LossFunction.h"
#include "../util/PA12TestsUtils.h"

int main(int argc, char* argv[]) {
    DataSet xorData =   DataSet("xor data", "./datasets/xor.txt");
    //test the numeric gradient calculations on a small
    //neural network
    testTinyGradientNumeric(xorData);

    //test the numeric gradient calculations on a small
    //neural network
    testSmallGradientNumeric(xorData);

    //test the numeric gradient calculations on a large 
    //neural network
    testLargeGradientNumeric(xorData);


    //this tests calculation of of the gradient via
    //the backwards pass for the small fully connected
    //neural network by comparing it to the
    //numeric gradient multiple times with random
    //starting weights
    testTinyGradients(xorData,  LossFunction::NONE);


    //this tests calculation of of the gradient via
    //the backwards pass for the small fully connected
    //neural network by comparing it to the
    //numeric gradient multiple times with random
    //starting weights
    testSmallGradients(xorData,  LossFunction::NONE);

    //this tests calculation of of the gradient via
    //the backwards pass for the large fully connected
    //neural network by comparing it to the
    //numeric gradient multiple times with random
    //starting weights
    testLargeGradients(xorData,  LossFunction::NONE);

    //this tests calculation of of the gradient via
    //the backwards pass for the tiny fully connected
    //neural network by comparing it to the
    //numeric gradient multiple times with random
    //starting weights
    testTinyGradientsMultiInstance(xorData,  LossFunction::NONE);


    //this tests calculation of of the gradient via
    //the backwards pass for the small fully connected
    //neural network by comparing it to the
    //numeric gradient multiple times with random
    //starting weights
    testSmallGradientsMultiInstance(xorData,  LossFunction::NONE);

    //this tests calculation of of the gradient via
    //the backwards pass for the large fully connected
    //neural network by comparing it to the
    //numeric gradient multiple times with random
    //starting weights
    testLargeGradientsMultiInstance(xorData,  LossFunction::NONE);
}