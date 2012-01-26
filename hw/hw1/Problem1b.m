% HW1 Q1-b
% =========================================================================
clear all;
load Data;

sample = [test_green; test_red];
training = [train_green; train_red];
group = [ones(n,1); zeros(n,1)];
k = [1 3 5 9 15 25 45 83 151];

% knn error for test data
knn_error_test = [];
for i = 1:9
% knnclassify is a matlab function
    class_knn_test = knnclassify(sample, training, group, k(i)); 
% count the number of points that are misclassified
    num_misclassified = sum(class_knn_test ~= [ones(m,1); zeros(m,1)]);
% calculate the test error by dividing points misclassified by total points 
    knn_error_test = [knn_error_test num_misclassified/(2*m)];  
end
figure(7); semilogx(200./k, knn_error_test, 'b-o'); hold on;

% knn error for training data
knn_error_train = [];
for i = 1:9
    class_knn_train = knnclassify(training, training, group, k(i));
    num_misclassified = sum(class_knn_train ~= [ones(n,1); zeros(n,1)]);
    knn_error_train = [knn_error_train num_misclassified/(2*n)]; 
end
semilogx(200./k, knn_error_train, 'g-o'); hold on;

% linear regression error
% classify is a matlab function for linear regression
class_linear_test = classify(sample, training, group); 
class_linear_train = classify(training, training, group);
% count the number of points that are misclassified
num_misclassified_test = sum(class_linear_test ~= [ones(m,1); zeros(m,1)]); 
num_misclassified_train = sum(class_linear_train ~= [ones(n,1); zeros(n,1)]);
% calculate the test error by dividing points misclassified by total points
linear_error_test =  num_misclassified_test/(2*m); 
linear_error_train =  num_misclassified_train/(2*n); 
% linear regression test data
semilogx(200./k, linear_error_test*ones(9,1), 'r--');hold on;

% linear regression train data
semilogx(200./k, linear_error_train*ones(9,1), 'm--');  hold on;
legend('k nearest neighbor test','k nearest neighbor train','lingear regression test','linear regression train');

xlabel('Degrees of Freedom - N/k'); ylabel('Test Error');
axis([1 300 0.15 0.25]);
hold off;
