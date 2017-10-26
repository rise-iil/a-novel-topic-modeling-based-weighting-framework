function prob_doc = run_PLSA (DATA)
% a runnable demo to show plsa in nlp application
MAXNUMDIM = 20000; % global variable, dimension of terms
MAXNUMDOC = 200000;  % global variable, number of documents
numTopic = 10;     % number of topics
numIter = 100;      % number of iteration

% 1th, preprocess the raw text set
termDocMatrix = (DATA(:,1:end-1))';

fprintf('Num of dimension: %d\n', size(termDocMatrix, 1));
fprintf('Num of document: %d\n', size(termDocMatrix, 2));

% 2th, fit a plsa model from a given term-doc matrix
% [prob_term_topic, prob_topic_doc, lls] = plsa(termDocMatrix, numTopic, numIter);
[prob_term_topic, prob_doc_topic, prob_topic] = plsa2(termDocMatrix, numTopic, numIter);

% compute the data distribution by marginalizing the P(d,z) on z.
prob_doc = sum((prob_doc_topic * diag(prob_topic)),2);

end 

