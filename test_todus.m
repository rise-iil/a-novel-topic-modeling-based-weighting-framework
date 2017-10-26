clc;
clear all;
close all;

file = 'datasets/cmc21.csv'; % Dataset
K = 5;
% Reading training file
data = dlmread(file);

% Performing 5-fold cross validation
indices = crossvalind('Kfold',length(data),K);
label = data(:,end);
for i=1:K
    test = (indices == i); train = ~test;
    train_data = data(train,:);
    test_data = data(test,:);
    test_op = data(test,size(test_data,2));
    % Calling TODUS with decision tree as the weak learner
	[prediction,score] = TODUS(train_data,test_data,'tree');
	% Storing the Precision and Recall Values
	output(i) = confusionmatStats(test_op,prediction);
	[~,~,~,auc(i)] = perfcurve(test_op,score(:,2),'1');
end