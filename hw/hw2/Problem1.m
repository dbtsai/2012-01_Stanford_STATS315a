% HW1 Q1
% Dong-Bang Tsai,Jin Chen, Wenqiong Guo
% =========================================================================
clear all;

M = 10;
mu_green = [0 1]; 
mu_red = [1 0];
sigma_green = [1 0; 0 1];
sigma_red = [1 0; 0 1];

% Generate M centroids from a bivariate Gaussian distribution
green_centroids = mvnrnd(mu_green,sigma_green, M); % multivariate normal distribution
red_centroids = mvnrnd(mu_red,sigma_red, M);

% Pick an index at random with probability 1/10, and then generate N(m_k,
% I/5)
n = 100; %number of points for each class in training data
m = 5000; % # of points for each class in test data

% Generate the training data
train_green = [];
train_red = [];
for i = 1:n
    rand_index_g = randi(M);% randomly pick a centroid
    rand_index_r = randi(M); 
    center_green = green_centroids(rand_index_g,:); 
    center_red = red_centroids(rand_index_r,:);
    train_green(i,:) = mvnrnd(center_green,sigma_green/5, 1);% multivariate normal dist
    train_red(i,:) = mvnrnd(center_red,sigma_red/5, 1); 
end

% Generate the test data
test_green = [];
test_red = [];
for j = 1:m
    rand_index_g = randi(M);% randomly pick a centroid
    rand_index_r = randi(M); 
    center_green = green_centroids(rand_index_g,:); 
    center_red = red_centroids(rand_index_r,:);
    test_green(j,:) = mvnrnd(center_green,sigma_green/5, 1);% multivariate normal dist
    test_red(j,:) = mvnrnd(center_red,sigma_red/5, 1);
end

% Save the variables
save Data;

% Plot data points
figure(1); plot(train_green(:,1), train_green(:,2),'go'); hold on;
plot(train_red(:,1), train_red(:,2),'ro'); 
% Linear regression for training data
min_x = min([train_green(:,1); train_red(:,1)]);
max_x = max([train_green(:,1); train_red(:,1)]);
min_y = min([train_green(:,2); train_red(:,2)]);
max_y = max([train_green(:,2); train_red(:,2)]);
[X,Y] = meshgrid(linspace(min_x, max_x),linspace(min_y, max_y)); % just to make data in correct format for 'classify'
X = X(:); Y = Y(:);
sample = [X Y];
training = [train_green; train_red];
group = [ones(n,1); zeros(n,1)];% green = 1, red = 0
[class,err,POSTERIOR,logp,coeff_linear] = classify(sample, training, group); % linear classifier
K = coeff_linear(1,2).const;
L = coeff_linear(1,2).linear; 
f = @(x,y) K + [x y]*L; % linear function from classifier
h = ezplot(f,[min_x max_x min_y max_y]);
title('Linear regression classifier')
set(h, 'Color', 'm', 'linewidth', 2); xlabel('x'); ylabel('y');
hold off;

figure(2); plot(test_green(:,1), test_green(:,2),'go'); hold on;
plot(test_red(:,1), test_red(:,2),'ro'); xlabel('x'); ylabel('y');

hold off;

% Plot KNN classifier for training data
for i = 1:4
    k = [1 5 25 151];
    figure(i+2); plot(train_green(:,1), train_green(:,2),'go'); hold on;
    plot(train_red(:,1), train_red(:,2),'ro');
    title(sprintf('%d of Nearest Neighbor Classifier',k(i)));
    % Knn classifier for training data (lacks error)
    class_knn = knnclassify(sample, training, group, k(i)); % knn classifier
    contour(linspace(min_x, max_x),linspace(min_y, max_y), reshape(class_knn, 100, 100), 1, 'b', 'linewidth', 2); %plot
    axis([min_x max_x min_y max_y]); xlabel('x'); ylabel('y');
    hold off;
end