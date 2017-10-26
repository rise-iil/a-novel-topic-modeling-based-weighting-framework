function [pred,score] = TODUS (TRAIN,TEST,WeakLearn)
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
% PLSA function call on D_maj (Negative Data)
prob_doc = run_PLSA(TRAIN);
prob_doc_maj = prob_doc(pos_size+1:end);
prob_doc_min = prob_doc(1:pos_size);

% Min-Max Normalizing the P(D_maj)
max_val = max(prob_doc_maj);
min_val = min(prob_doc_maj);
W_maj = (prob_doc_maj - min_val) / (max_val - min_val);

% Normalizing W_maj
prob_doc_maj = W_maj./ sum(W_maj);
 
% Converting training set into Weka compatible format
CSVtoARFF (TRAIN, 'train', 'train');
train_reader = javaObject('java.io.FileReader', 'train.arff');
train = javaObject('weka.core.Instances', train_reader);
train.setClassIndex(train.numAttributes() - 1);
    
% Making negative data, same size as of POS_DATA
neg_equal_size = pos_size;

% Undersampling the NEG_DATA based on W_maj
RESAM_NEG = NEG_DATA(randsample(length(NEG_DATA),neg_equal_size,true,prob_doc_maj),:);

RESAMPLED = [POS_DATA;RESAM_NEG];
  
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