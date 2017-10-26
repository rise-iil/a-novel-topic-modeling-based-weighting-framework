function [prob_term_topic, prob_doc_topic, prob_topic,prob_topic_term_doc] = plsa2(termDocMatrix, numTopic, iter)

[numTerm, numDoc] = size(termDocMatrix);

prob_topic = rand(numTopic, 1); % p(topic)
prob_topic = prob_topic./sum(prob_topic); % normalization

prob_term_topic = rand(numTerm, numTopic); % p(term | topic)
for z = 1:numTopic
	prob_term_topic(:, z) = prob_term_topic(:, z) / sum(prob_term_topic(:, z)); % normalization
end

prob_doc_topic = rand(numDoc, numTopic);   % p(doc | topic)
for z = 1:numTopic
	prob_doc_topic(:, z) = prob_doc_topic(:, z) / sum(prob_doc_topic(:, z)); % normalization
end
prob_topic_term_doc = cell(numTopic, 1);   % p(topic | doc, term)

for z = 1 : numTopic
	prob_topic_term_doc{z} = zeros(numTerm, numDoc);
end %

for i = 1 : iter
    fprintf('Iteration %d\n', i);
    for z = 1:numTopic
		prob_topic_term_doc{z} = (prob_term_topic(:, z) * prob_doc_topic(:, z)') * prob_topic(z);
    end
    
    C = 0;
    for z = 1:numTopic
        C = C+prob_topic_term_doc{z}(:,:);
    end
    
    for z = 1:numTopic
        prob_topic_term_doc{z} = prob_topic_term_doc{z}./C;
    end

    for z = 1:numTopic
		prob_doc_topic(:, z) = sum(termDocMatrix .* prob_topic_term_doc{z});
		prob_topic(z) = sum(prob_doc_topic(:, z));
		prob_doc_topic(:, z) = prob_doc_topic(:, z) / prob_topic(z); % normalization
        assert((sum(prob_doc_topic(:, z)) - 1.0) < 1e-3);
    end
	for z = 1:numTopic
		prob_term_topic(:, z) = sum(termDocMatrix .* prob_topic_term_doc{z}, 2);
        assert(prob_topic(z) - sum(prob_term_topic(:, z)) < 1e-6);
		prob_term_topic(:, z) = prob_term_topic(:,z) / sum(prob_term_topic(:,z)); % normalization
        assert((sum(prob_term_topic(:, z)) - 1.0) < 1e-6)
	end
	
	prob_topic(:) = prob_topic(:) / sum(prob_topic(:)); % normalization	
    assert((sum(prob_topic(:)) - 1.0) < 1e-6);
end
save model.mat prob_doc_topic prob_term_topic prob_topic
end