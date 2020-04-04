function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
m = length(y); 
J = 0;
grad = zeros(size(theta));

J = (-1/m)*(y'*log(sigmoid(X*theta)) + (1-y)'*log(1-sigmoid(X*theta))) + lambda*sum(theta(2:end).^2)/m;
grad(1) = X(:,1)'*(sigmoid(X*theta) - y)/m;
grad(2:end) = X(:,2:end)'*(sigmoid(X*theta) - y)/m + lambda*theta(2:end)/m;
end
