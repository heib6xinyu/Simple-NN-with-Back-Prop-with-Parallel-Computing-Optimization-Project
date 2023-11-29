g++ data/DataSet.cpp data/Instance.cpp network/*.cpp util/*.cpp BasicTests.cpp -o BasicTests -std=c++11
g++ data/DataSet.cpp data/Instance.cpp network/*.cpp util/*.cpp NNTests.cpp -o NNTests -std=c++11
g++ data/DataSet.cpp data/Instance.cpp network/*.cpp util/*.cpp GradientTests.cpp -o GradientTests -std=c++11
g++ data/DataSet.cpp data/Instance.cpp network/*.cpp util/*.cpp GradientDescent.cpp -o GradientDescent -std=c++11