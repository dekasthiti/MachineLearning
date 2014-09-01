function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(nn_params));
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Modify X to include the bias node
X = [ ones( m, 1 ) X ];
A = sigmoid( Theta1 * X' );
A = [ ones(1, m); A];
H_Theta = sigmoid( Theta2 * A );

% Map predicted values in y to classified values in Y
Y       = zeros( num_labels, m );

for i = 1:m
	Y(y(i), i) = 1;
end

cost = Y.*log(H_Theta) + (1 - Y).*log(1-H_Theta);
 
J = -(sum(cost(:)))/m;

%========================================================================%
%Regularized cost function

Theta1_square = Theta1(:,2:end).^2;
Theta2_square = Theta2(:,2:end).^2;

J = J + lambda/(2*m) * (sum(Theta1_square(:)) + sum(Theta2_square(:)));

%========================================================================%
			

%========================================================================%
%Back Propagation for gradient determination
%========================================================================%

a1 = zeros( input_layer_size+1, 1 );  %a1 is a column vector
a2 = zeros( hidden_layer_size, 1 ); %a2 is a column vector
a3 = zeros( num_labels, 1);			%a3 is a column vector
a1_prime = zeros( input_layer_size + 1, 1);
a2_prime = zeros( hidden_layer_size + 1, 1);

delta3   = zeros( num_labels, 1);
delta2   = zeros( hidden_layer_size+1 , 1 );

for i = 1:m
	
	a1 = [X( i, :)]';			%401 x 1
	a1_prime = a1;			    %401 x 1
	z2 = Theta1 * a1_prime;     %25x401 * 401x1 = 25x1
	a2  = sigmoid( z2 );		%25x1
	a2_prime = [1; a2];			%26x1
	a2_prime = [1; a2];			%26x1
	a3 = sigmoid( Theta2 * a2_prime );  %10x26 * 26x1 = 10x1
	
	delta3 = a3 - Y(:, i);      %10x1
	Theta2_grad = Theta2_grad + delta3 * a2_prime';
	
	%z2 = [ 1; z2];
	delta2 = (Theta2' * delta3).* (a2_prime.*(1-a2_prime));%sigmoidGradient(z2);		
	Theta1_grad = Theta1_grad + delta2(2:end) * a1_prime';
end

Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

%Gradient with regularization
Theta1_grad(:,2:end ) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
