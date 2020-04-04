function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
m = size(X, 1); 
p = zeros(m, 1);
p = sigmoid(X*theta);
p(p >= 0.5) = 1;
p(p < 0.5) = 0;
end
