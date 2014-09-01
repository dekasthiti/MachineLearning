function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

%local variables
diff = 0;
sum_of_squares = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
for i = 1: m

	h_theta(i) = theta(1) * X(i, 1) + theta(2) * X(i, 2);
	diff = ( h_theta(i) - y(i) );
	sum_of_squares = (diff * diff) + sum_of_squares;
end

J = ( sum_of_squares/ (2 * m) );

% =========================================================================

end
