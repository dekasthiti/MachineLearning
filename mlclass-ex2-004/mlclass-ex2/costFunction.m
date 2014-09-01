function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

%=================================================================================
%This section of the code is written using loops. It's accuracy has been verified
%==================================================================================
h_theta = zeros(m, 1);
cost = zeros(m, 1);

for i = 1:m
	h_theta(i) = sigmoid([X(i,:)] * theta);
	cost(i) =(y(i) * log( h_theta(i) ) ) + ( (1 - y(i) ) * log(1-h_theta(i) ) );	
end
J = -sum(cost(:))/m;

error = zeros(m,1);
error = h_theta - y;
X_temp = zeros(size(X));
for j= 1:length(theta)
	for k = 1:m
		X_temp(k,j) = error(k,1) * X(k,j);
	end
	grad(j) = (1/m) * sum(X_temp(:,j));
end	
%======================================================================================


%======================================================================================
%This part of the code is vectorized. It's accuracy is being verified
%======================================================================================
% H_Theta = zeros(m, 1);
% cost    = zeros(m, 1);

% H_Theta = sigmoid( X * theta );

% cost =  y.*log(H_Theta) + (1 - y).*log(1-H_Theta); 

% J = -1/m * sum(cost(:));
% grad = 1/m * X' * (H_Theta - y);

% =============================================================
end
