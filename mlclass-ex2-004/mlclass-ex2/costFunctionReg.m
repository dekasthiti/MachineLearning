function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


[J grad]= costFunction(theta, X, y);

%===================================================================================
%This section of the code is written using for loop. It's accuracy has been verified
%====================================================================================
for j= 1:length(theta)
	if( j > 1) % we do not regularize the first element in the gradient vector
		grad(j) = grad(j) + (lambda/m) * theta(j);
		theta_square(j) = theta(j) * theta(j);
	end
end	

J = J + (lambda/(2*m)) * sum(theta_square(:));
%======================================================================================

% =============================================================

%====================================================================
%This part of the code is vectorized. It's accuracy is being verified
%=====================================================================
% theta_square = zeros(size(theta) - 1);
% theta_square = theta(2:end).^2;

% J = J + (lambda/(2*m)) * sum(theta_square(:));
% grad(2:end) = grad(2:end) + (lambda/m ) * theta(2:end);
%=====================================================================

end
