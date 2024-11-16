%% Noah Plant
% 11/15/2024

%% Purpose
% The purpose of this script is to create a deep neural network to predict
% future temperature from various weather data. 



layers = [
    sequenceInputLayer(9,"Name","sequence")
    lstmLayer(128,"Name","lstm")
    fullyConnectedLayer(9,"Name","fc")
    swishLayer("Name","swish")
    fullyConnectedLayer(9,"Name","fc_1")
    swishLayer("Name","swish_1")
    fullyConnectedLayer(1,"Name","fc_2")];


%net=addLayers(net,regressionLayer);

% clean up helper variable
%clear tempNet;
%net = initialize(net);
%lgraph = layerGraph(net);

options = trainingOptions("adam","Verbose",true,"VerboseFrequency", ...
    10,"MiniBatchSize",8, ...
    "InitialLearnRate",1e-4,"ValidationPatience",20,"ValidationData", ...
    {XtrainRED,TtrainSTD}, ...
    "MaxEpochs",2000,"Plots","training-progress","Shuffle","every-epoch");

trainednet = trainnet(XtrainRED,TtrainSTD,layers,"mse",options);
