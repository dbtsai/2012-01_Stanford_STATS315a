% H1 Q1-c
% =========================================================================
clear all;
load Data;

[n, m] = size(train_green);
training = [train_green; train_red];
k = [1 3 5 9 15 25 45 83 151]; % k values for knn classification

knn_avg_error = []; % initialize the errors
linear_avg_error = [];
for i = 1:10 % loop over each 10% chunk of the data for 10-fold cross validation
    training_10 = [train_green(int8(0.1*(i-1)*n+1):int8(0.1*i*n),:); train_red(int8(0.1*(i-1)*n+1):int8(0.1*i*n),:)]; % the 10% of the data
    training_g_90 = train_green;
    training_r_90 = train_red;
    training_g_90(int8(0.1*(i-1)*n+1):int8(0.1*i*n),:) = [];
    training_r_90(int8(0.1*(i-1)*n+1):int8(0.1*i*n),:) = [];
    training_90 = [training_g_90; training_r_90]; % the rest 90% of the data
    group = [ones(0.9*n,1); zeros(0.9*n,1)];

    % KNN classification
    knn_error = [];
    for i = 1:9
        class_knn = knnclassify(training_10, training_90, group, k(i));
        num_misclassified = sum(class_knn ~= [ones(0.1*n,1); zeros(0.1*n,1)]);
        knn_error = [knn_error num_misclassified/(2*0.1*n)]; 
    end
    knn_avg_error = [knn_avg_error; knn_error]; % keep a running vector of all the errors
    
    % Linear regression
    linear_error = [];
    class_linear = classify(training_10, training_90, group);
    num_misclassified = sum(class_linear ~= [ones(0.1*n,1); zeros(0.1*n,1)]);
    linear_error =  num_misclassified/(2*0.1*n); 
    linear_avg_error = [linear_avg_error linear_error]; % keep a running vector of all the errors
end

knn_sd = std(knn_avg_error);
linear_sd = std(linear_avg_error);
knn_avg_error = mean(knn_avg_error); % average of all the errors for each fold
linear_avg_error = mean(linear_avg_error); % average of all the errors for each fold

figure(8); semilogx(200./k, knn_avg_error, 'b-o'); hold on;
plot(200./k, linear_avg_error*ones(9,1), 'r--');
xlabel('Degrees of Freedom - N/k'); ylabel('Average Errors');
title('10 fold cross-validation');
legend('k nearest neighbors','linear regression')
hold off;

figure(9); semilogx(200./k, knn_sd, 'b-o'); hold on;
plot(200./k, linear_sd*ones(9,1), 'r--');
xlabel('Degrees of Freedom - N/k'); ylabel('Standard Errors');
title('10 fold cross-validation');
legend('k nearest neighbors','linear regression')
hold off;
