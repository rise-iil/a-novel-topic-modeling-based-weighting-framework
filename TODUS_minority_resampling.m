function [pred,score] = TODUS_minority_resampling (TRAIN,TEST,WeakLearn,prob_doc)
% This function implements the TODUS Algorithm.
% Input: TRAIN = Training data as matrix
%        TEST = Test data as matrix
%        WeakLearn = String to choose algortihm. Choices are
%                    'svm','tree','knn' and 'logistic'.
% Output: prediction = size(TEST,1)x 1 vector. Col is class labels for 
%                      all instances.


javaaddpath('weka.jar');

%% Training
% Total number of instances in the training set
m = size(TRAIN,1);
POS_DATA = TRAIN(TRAIN(:,end)==1,:);
NEG_DATA = TRAIN(TRAIN(:,end)==-1,:);
pos_size = size(POS_DATA,1);
neg_size = size(NEG_DATA,1);
% Reorganize TRAIN by putting all the positive and negative exampels
% together, respectively.
TRAIN = [POS_DATA;NEG_DATA];
% Converting training set into Weka compatible format
CSVtoARFF (TRAIN, 'train', 'train');
train_reader = javaObject('java.io.FileReader', 'train.arff');
train = javaObject('weka.core.Instances', train_reader);
train.setClassIndex(train.numAttributes() - 1);
    
% Making negative data, same size as of POS_DATA

pos_indices = randsample(pos_size,pos_size,true,prob_doc(1:pos_size,:));
indices = 1:pos_size;
pos_indices2 = setdiff(indices,pos_indices);
pos_indices = [pos_indices;pos_indices2'];

RESAM_POS = POS_DATA(pos_indices,:);
RESAM_NEG = NEG_DATA(randsample(neg_size,length(pos_indices),true,prob_doc(pos_size+1:end,:)),:);

RESAMPLED = [RESAM_POS;RESAM_NEG];

% Converting resample training set into Weka compatible format
CSVtoARFF (RESAMPLED,'resampled','resampled');
reader = javaObject('java.io.FileReader','resampled.arff');
resampled = javaObject('weka.core.Instances',reader);
resampled.setClassIndex(resampled.numAttributes()-1);  
    
% Training a weak learner. 
switch WeakLearn
    case 'svm'
        model = javaObject('weka.classifiers.functions.SMO');
    case 'tree'
        model = javaObject('weka.classifiers.trees.J48');
    case 'knn'
        model = javaObject('weka.classifiers.lazy.IBk');
        model.setKNN(5);
    case 'logistic'
        model = javaObject('weka.classifiers.functions.Logistic');
end
model.buildClassifier(resampled);

n = size(TEST,1); % Total number of instances in the test set

CSVtoARFF(TEST,'test','test');
test = 'test.arff';
test_reader = javaObject('java.io.FileReader', test);
test = javaObject('weka.core.Instances', test_reader);
test.setClassIndex(test.numAttributes() - 1);
pred = zeros(n,1);
for i = 0 : n-1
    pred(i+1) = model.classifyInstance(test.instance(i));
    score(i+1,:) = model.distributionForInstance(test.instance(i));
    if pred(i+1) == 0
        pred(i+1) = -1;
    end
end