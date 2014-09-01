function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

disp( m);
disp(J_history);
disp(theta);
disp(alpha);
disp(num_iters);

% diff_all    = zeros(length(y), 1);
	% prod_theta_1(:)= zeros(length(y), 1);
	% prod_theta_2(:)= zeros(length(y), 1);
	% x_1(:)= zeros(length(y), 1);
	% x_2(:)= zeros(length(y), 1);
	% sum_theta_1 = 0;
	% sum_theta_2 = 0;

% h_theta = X * theta;
% diff_all = h_theta - y;
% x_1      = X(:,1);
% x_2      = X(:,2);
% prod_theta_1 = (diff_all).* x_1;
% prod_theta_2 = (diff_all).* x_2;
% sum_theta_1  = cumsum(prod_theta_1)(m);
% sum_theta_2  = cumsum(prod_theta_2)(m);

for iter = 1:num_iters

	
	h_theta     = zeros(length(y), 1);
	diff        = 0;
	prod_theta_1= 0;
	prod_theta_2= 0;
	sum_theta_1 = 0;
	sum_theta_2 = 0;
	temp1       = 0;
	temp2       = 0;
	
	for i = 1:m
		h_theta(i) = theta(1) * X(i, 1) + theta(2) * X(i, 2);
		diff       = h_theta(i) - y(i);
		prod_theta_1 = diff * X(i, 1);
		prod_theta_2 = diff * X(i, 2);
		sum_theta_1 = prod_theta_1 + sum_theta_1;
		sum_theta_2 = prod_theta_2 + sum_theta_2;
	end
	
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %	
	temp1 = theta(1) - (alpha/m)* sum_theta_1;
	temp2 = theta(2) - (alpha/m)* sum_theta_2;
	
	theta(1) = temp1;
	theta(2) = temp2;

    fprintf('%f %f \n', theta(1), theta(2));

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);	
    fprintf('%f \n', J_history(iter));


end

end
