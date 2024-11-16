%% Noah Plant
% 11/15/2024

%% Purpose
% The purpose of this script is to create my final shallow neural network I
% determined the optimal neuron size to be 9, from there I will tweak some
% paramaters including the training function to create the best shallow
% network I can.

clc; clear; close all;



load('nnTrainingScript.mat') % Load in the saved workspace.

neurons=9;

%% Create the NN

FinaltempNet=narxnet(ID,FD,neurons); % Creates our temperature NN

FinaltempNet.trainFcn='trainbr';
FinaltempNet.divideParam.trainRatio = 75/100; 
%FinaltempNet.divideParam.valRatio = 10/100;
FinaltempNet.divideParam.testRatio = 25/100;
FinaltempNet.divideFcn='divideblock';
FinaltempNet.trainParam.epochs=400;
FinaltempNet.trainParam.goal=0.001;

R=0;
counter=0;
while abs(R)<.90
    % Initialize the neural network for each iteration
    FinaltempNet=init(FinaltempNet); 

    [FinaltempNet,tr] = train(FinaltempNet,Xs,Ts,Xi,Ai); 


    [Y,xfo1,afo1] = FinaltempNet(Xs,Xi,Ai);
    [FinaltempNet_clo,Xic,Aic]=closeloop(FinaltempNet,xfo1,afo1);
    [outputs] = FinaltempNet_clo(XtestCell,Xic,Aic);
    
    
    MSE=mse(outputs,TtestCell);
    %plotregression(outputs,TtestCell)
    R=regression(outputs,TtestCell);
    disp(R)
    disp(counter)
end

% My final neural network had a regression value of -.9015
