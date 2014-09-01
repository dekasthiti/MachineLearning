function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_train = [.01 .03 .1 .3 1 3 10 30];
sigma_train   = [.01 .03 .1 .3 1 3 10 30];

prediction_error = zeros(length(C_train), length(sigma_train));
predictions = zeros(length(yval),1);

for i = 1:length(C_train)
	for j = 1:length(sigma_train)
		model= svmTrain(X, y, C_train(i), @(x1, x2) gaussianKernel(x1, x2, sigma_train(j)));
		predictions = svmPredict( model, Xval);
		prediction_error(i,j) = mean(double(predictions ~= yval));
		
	end
end

[MinimumForEveryC sigma_indices] = min(prediction_error, [], 2);

[Global_min C_index] = min(MinimumForEveryC);

C = C_train(C_index);
sigma = sigma_train(sigma_indices(C_index));



% =========================================================================

end
