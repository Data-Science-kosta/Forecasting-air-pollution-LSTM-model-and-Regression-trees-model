function [Xout_train, Xout_test] = FeatureScaling(Xtrain, Xtest)
[~, n] = size(Xtrain);
Xout_train = zeros(size(Xtrain));
Xout_test = zeros(size(Xtest));
for i = 1:n
   mean_train = mean(Xtrain(:,i));
   std_train(i) = std(Xtrain(:,i));
   Xout_train(:,i) = (Xtrain(:,i) - mean_train)./std_train(i); 
   mean_test = mean(Xtest(:,i));
   std_test(i) = std(Xtest(:,i));
   Xout_test(:,i) = (Xtest(:,i) - mean_test)./std_test(i);
end
Xout_train(:,std_train < 0.0001) = []; % removing features that doesnt change thrugh exampes
Xout_test(:,std_train < 0.0001) = []; % (namerno sam stavio std_train)
end