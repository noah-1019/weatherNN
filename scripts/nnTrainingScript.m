%% Noah Plant 
% 11/10/2024

% The purpose of this script is to predict temperature using a time series
% of varius weather inputs. The inputs were collected over the space of 1
% year every 10 minutes totally around 56,000 datapoints. 

clear; clc; close all

%% Load in and format data

% Load in the data
tempData = readtable('cleaned_weather.csv');
tempData2=tempData(:,2:end); % Data without date column

colNames=tempData2.Properties.VariableNames;
featurenames=colNames(:,1:17);

% Convert data to numeric input and target values.
dataMatrix=table2array(tempData2);


% Targets
T=[dataMatrix(:,2)];
    % Column 1: Air Temperature

% Inputs
X=[dataMatrix(:,1),dataMatrix(:,4:end-1)];

% Segment data into training and testing components
    % Training:90% 
    % Testing: 10%

targetSize=size(X);
trainingIndex=round(targetSize(1)*.9);

Xtrain=X(1:trainingIndex,:);
Xtest=X(trainingIndex+1:end,:);

Ttrain=T(1:trainingIndex,:);
Ttest=T(trainingIndex+1:end,:);

% Normalize data

    % Normalize training data

mean_input_train=mean(Xtrain); % Takes the mean for each factor
std_input_train=std(Xtrain); % Takes the std for each factor

mean_target_train=mean(Ttrain); % Takes the mean for each factor
std_target_train=std(Ttrain); % Takes the std for each factor

XtrainSTD= (Xtrain-mean_input_train)./std_input_train; 
TtrainSTD=(Ttrain-mean_target_train)./std_target_train;

    % Normalize test data

mean_input_test=mean(Xtest); % Takes the mean for each factor
std_input_test=std(Xtest); % Takes the std for each factor

mean_target_test=mean(Ttest); % Takes the mean for each factor
std_target_test=std(Ttest); % Takes the std for each factor

XtestSTD= (Xtest-mean_input_test)./std_input_test; 
TtestSTD=(Ttest-mean_target_test)./std_target_test;

%% Perform PCA
% The purpose of this segment is to reduced the dimensionality of my
% dataset. I must do quite a bit of reduction because I do not have the
% processing power to intensivly train a NN.

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(XtrainSTD);
[COEFFT, SCORET, LATENTT, TSQUAREDT, EXPLAINEDT] = pca(XtestSTD);

explainedsum = cumsum(EXPLAINED); % cumulatively the explained variance
numCom=find(explainedsum>95,1,"first"); % Number of components that explain 95 percent of the data
COEFF_CLIP=COEFF(:,numCom);
EXPLAINED_CLIP=EXPLAINED(1:8)';
featureImportance=sum((COEFF'.^2.*EXPLAINED),2);

% This creates a graph that shows the order of the feature importance.
bar(featurenames,featureImportance)
title("Principle Component Analysis Results")
% Using PCA I have reduced the number of components from 16 to 9, this
% should improve the speed in which I can train my NN. Further reduction
% may be required as I have 56,000 datapoints and I am running this script
% on my crappy laptop. These 9 components account for 95% of the data.
XtrainRED=SCORE(:,1:numCom);
XtestRED=SCORET(:,1:numCom);

% Clean Workspace up:

clear("targetSize","tempData","tempData2","trainingIndex","mean_input_test", ...
    "std_input_test","mean_input_train","std_input_train","dataMatrix","X","Xtest","XtestSTD","XtrainSTD", ...
    "numCom","colNames","featureNames","Xtrain","T","Ttrain","Ttest","mean_target_test", ...
    "mean_target_train","explainedsum")

%% Train time series NN

% Get data in cell format for the time series NN
TtestCell=tonndata(TtestSTD,0,0);
TtrainCell=tonndata(TtrainSTD,0,0);
XtestCell=tonndata(XtestRED,0,0);
XtrainCell=tonndata(XtrainRED,0,0);


% Initialize FD ID and the neuron number.
FD=(2:6);
ID=(0:20);
neurons=10;

tempNet=narxnet(ID,FD,neurons); % Creates our temperature NN

[Xs,Xi,Ai,Ts] = preparets(tempNet,XtrainCell,{},TtrainCell);







%% Optimize Neuron Number

mseMatrix=ones(10,3);

for i=1:10 % 10 neurons
    for j=1:3 % 5 trials for each neuron
        disp("Neurons \n")
        disp(i)
        disp("Trials \n")
        disp(j)

        tempNet=narxnet(ID,FD,neurons); % Creates our temperature NN

        tempNet.divideParam.trainRatio = 75/100; 
        tempNet.divideParam.valRatio = 10/100;
        tempNet.divideParam.testRatio = 15/100;
        tempNet.divideFcn='divideblock';
        tempNet.trainParam.epochs=500;
        tempNet.trainParam.goal=0.001;

        
        % Initialize the neural network for each iteration
        tempNet=init(tempNet); 

        [tempNet,tr] = train(tempNet,Xs,Ts,Xi,Ai); 


        [Y,xfo1,afo1] = tempNet(Xs,Xi,Ai);
        [tempNet_clo,Xic,Aic]=closeloop(tempNet,xfo1,afo1);
        [outputs] = tempNet_clo(XtestCell,Xic,Aic);


        mseMatrix(i,j)=mse(outputs,TtestCell);
    end


end


% The end result was 9 neurons to be optimal with a mean mse of 3.8102



%% Saving Data
% The purpose of this script was just to format the data, reduce the dimensionality
% of my dataset, and determine the optimal neural network. I will now save
% my data so that I can create a neural network with 9 neurons in a
% different script.








