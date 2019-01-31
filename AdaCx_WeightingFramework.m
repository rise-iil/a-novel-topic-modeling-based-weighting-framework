function [prediction,score] = AdaCx_WeightingFramework (TRAIN,TEST,WeakLearn,prob_doc,Type)

% Input: TRAIN = Training data as matrix
%        TEST = Test data as matrix
%        WeakLearn = String to choose algortihm. Choices are
%                    'svm','tree','knn' and 'logistic'.
%        Costs_i \in (0,1]
%        Type = 1/2/3 for AdaC1/AdaC2/AdaC3 modules respectively
% Output: prediction = size(TEST,1)x 2 matrix. Col 1 is class labels for
%                      all instances. 

javaaddpath('weka.jar');

%% Training
% Total number of instances in the training set
m = size(TRAIN,1);
POS_DATA = TRAIN(TRAIN(:,end)==1,:);
NEG_DATA = TRAIN(TRAIN(:,end)==-1,:);
pos_size = size(POS_DATA,1);
neg_size = size(NEG_DATA,1);

% Calculation of Imbalance Ratio
imbalance_ratio_constant = 1;
imbalance_ratio = (pos_size/neg_size)*imbalance_ratio_constant;

% Reorganize TRAIN by putting all the positive and negative exampels
% together, respectively.
TRAIN = [POS_DATA;NEG_DATA];


% Converting training set into Weka compatible format
CSVtoARFF (TRAIN, 'train', 'train');
train_reader = javaObject('java.io.FileReader', 'train.arff');
train = javaObject('weka.core.Instances', train_reader);
train.setClassIndex(train.numAttributes() - 1);

% Total number of iterations of the boosting method
T = 20;

% Initial weight assignment
W = zeros(1,m);
minval_minority = min(prob_doc(1:pos_size));
maxval_minority = max(prob_doc(1:pos_size));
minval_majority = min(prob_doc(pos_size+1:end));
maxval_majority = max(prob_doc(pos_size+1:end));
prob_doc(1:pos_size) = (prob_doc(1:pos_size)-minval_minority)./(maxval_minority-minval_minority);
prob_doc(pos_size+1:end) = (prob_doc(pos_size+1:end)-minval_majority)./(maxval_majority-minval_majority);

for i = 1:m
    W(1,i) = prob_doc(i);
end

sum_W = sum(W(1,:));

for i = 1:m
    W(1,i) = W(1,i)/sum_W;
end

%Initial Cost Assignment
costs = prob_doc;
costs(1:pos_size) = prob_doc(1:pos_size);
costs(pos_size+1:end) = prob_doc(pos_size+1:end).*imbalance_ratio;

% Total number of iterations of the boosting method
T = 20;

% L stores pseudo loss values, H stores hypothesis, AlphaT stores 0.5*log(1/beta)
% values that is used as the weight of the % hypothesis while forming the
% final hypothesis. % All of the following are of length <=T and stores
% values for every iteration of the boosting process.
L = [];
H = {};
AlphaT=[];

% Loop counter
t = 1;

% Keeps counts of the number of times the same boosting iteration have been
% repeated
count = 0;

% create a random generator object.
random = javaObject('java.util.Random');

% Boosting T iterations
while t <= T

    % LOG MESSAGE
    disp (['Boosting iteration #' int2str(t)]);

    % Training a weak learner. 'pred' is the weak hypothesis. However, the
    % hypothesis function is encoded in 'model'.
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
    
    % perform random sampling based on the document weights.
    train_sampled = train.resampleWithWeights(random, W(t,:));

    % build the classifier model with the training data.
    model.buildClassifier(train_sampled);

    pred = zeros(m,1);
    for i = 0 : m - 1
        pred(i+1) = model.classifyInstance(train.instance(i));
        if pred(i+1) == 0
            pred(i+1) = -1;
        end
    end

    % Computing the pseudo loss of hypothesis 'model'
    loss = 0;
    for i = 1:m
        if TRAIN(i,end)==pred(i)
            continue;
        else
            loss = loss + W(t,i);
        end
    end
    
    if loss < 0.000001
        disp('break due to NaN');
        break;
    end

    fprintf( 1, 'loss at iteration: %d => loss: %f\n', t, loss );

    % If count exceeds a pre-defined threshold (5 in the current
    % implementation), the loop is broken and rolled back to the state
    % where loss > 0.5 was not encountered.
    if count > 5
       L = L(1:t-1);
       H = H(1:t-1);
       AlphaT=AlphaT(1:t-1);
       disp ('Too many iterations have loss > 0.5');
       disp ('Aborting boosting...');
       break;
    end

    % If the loss is greater than 1/2, it means that an inverted
    % hypothesis would perform better. In such cases, do not take that
    % hypothesis into consideration and repeat the same iteration. 'count'
    % keeps counts of the number of times the same boosting iteration have
    % been repeated
    if loss > 0.5
        count = count + 1;
        continue;
    else
        count = 1;
    end

    L(t) = loss; % Pseudo-loss at each iteration
    H{t} = model; % Hypothesis function
    beta = loss/(sum(W(t,:))-loss); % Setting weight update parameter 'beta'.
    AlphaT(t) = 0.5*log(1/beta); % Weight of the hypothesis

    % At the final iteration there is no need to update the weights any
    % further
    if t==T
        break;
    end
    
    if (Type==1)
        Outer = ones(m,1);
        Inner = costs;
    elseif (Type==2)
        Inner = ones(m,1);
        Outer = costs;
    elseif (Type==3)
        Inner = costs;
        Outer = costs;
    end

    % Updating weight
    for i = 1:m
        W(t+1,i) = Outer(i)*W(t,i)*exp(-AlphaT(t)*Inner(i)*pred(i)*TRAIN(i,end));
    end

    % Normalizing the weight for the next iteration
    sum_W = sum(W(t+1,:));
    for i = 1:m
        W(t+1,i) = W(t+1,i)/sum_W;
    end
    % Incrementing loop counter
    t = t + 1;
end

% The final hypothesis is calculated and tested on the test set
% simulteneously.

%% Testing RUSBoost
n = size(TEST,1); % Total number of instances in the test set

CSVtoARFF(TEST,'test','test');
test = 'test.arff';
test_reader = javaObject('java.io.FileReader', test);
test = javaObject('weka.core.Instances', test_reader);
test.setClassIndex(test.numAttributes() - 1);

prediction = zeros(n,1);
score = zeros(n,1);

for i = 1:n
    sumP = 0;
    count = 0;
    for j = 1:size(H,2)
        p = H{j}.classifyInstance(test.instance(i-1));
	% predictions are 1 & 0, so change that to +1, -1
        if p == 0
            p = -1;
        else
            count = count+1;
        end
        sumP = sumP + AlphaT(j)*p;
    end
    score(i) = count/size(H,2);
    if (sumP>0)
        prediction(i) = 1;
    else
        prediction(i) = -1;
    end
end

