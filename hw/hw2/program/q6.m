% HW2 Q6
% Dong-Bang Tsai
% =========================================================================
clear all;

% Generate training set
zip_train_raw_3 = dlmread('train.3'); 
zip_train_raw_5 = dlmread('train.5'); 
zip_train_raw_8 = dlmread('train.8'); 
X = [zip_train_raw_3; zip_train_raw_5; zip_train_raw_8];
Y = [3*ones(size(zip_train_raw_3,1),1); ...,
     5*ones(size(zip_train_raw_5,1),1); 8*ones(size(zip_train_raw_8,1),1)];
clear zip_train_raw_3 zip_train_raw_5 zip_train_raw_8;

% Generate testing set
zip_test_raw = dlmread('zip.test'); 
zip_test_raw_3 = zip_test_raw( find(zip_test_raw(:,1)==3), 2:257);
zip_test_raw_5 = zip_test_raw( find(zip_test_raw(:,1)==5), 2:257);
zip_test_raw_8 = zip_test_raw( find(zip_test_raw(:,1)==8), 2:257);
X_test = [zip_test_raw_3; zip_test_raw_5; zip_test_raw_8];
Y_test = [3*ones(size(zip_test_raw_3,1),1); ...,
     5*ones(size(zip_test_raw_5,1),1); 8*ones(size(zip_test_raw_8,1),1)];
clear zip_test_raw zip_test_raw_3 zip_test_raw_5 zip_test_raw_8;


% a) 
% The error of training set:
error_train = 100*sum(classify(X, X, Y,'linear')~=Y)/length(Y)
% The error of testing set:
error_test  = 100*sum(classify(X_test, X, Y,'linear')~=Y_test)/length(Y_test)

% b)
[U,S,V] = svd(X);
V_leading = V(:,1:49);
clear U S V;
X_leading = X*V_leading;
X_test_leading = X_test*V_leading;
% The error of training set:
error_leading_train = 100*sum(classify(X_leading, X_leading, Y,'linear') ...,
                      ~=Y)/length(Y)
% The error of testing set:
error_leading_test  = 100*sum(classify(X_test_leading, X_leading, Y,'linear') ...,
                        ~=Y_test)/length(Y_test)

% c)
X_square=reshape(X,length(Y),16,16);
X_test_square=reshape(X_test,length(Y_test),16,16);
% Average the non-operlapping 2x2 pixel blocks
X_square_filtered = zeros(length(Y),8,8);
X_test_square_flitered = zeros(length(Y_test),8,8);
for i=1:8
    for j=1:8
        X_square_filtered(:,i,j) = (X_square(:,2*i-1,2*j-1) +X_square(:,2*i,2*j-1) ...,
                                    +X_square(:,2*i-1,2*j) +X_square(:,2*i,2*j))/4;
        X_test_square_flitered(:,i,j) = (X_test_square(:,2*i-1,2*j-1) ...,
                           +X_test_square(:,2*i,2*j-1) +X_test_square(:,2*i-1,2*j) ...,
                           +X_test_square(:,2*i,2*j))/4;
    end
end
X_filtered = reshape(X_square_filtered, length(Y), 64);
X_test_filtered = reshape(X_test_square_flitered, length(Y_test), 64);
clear X_square X_test_square X_square_filtered X_test_square_flitered;
% The error of training set:
error_filtered_train = 100*sum(classify(X_filtered, X_filtered, Y,'linear') ...,
                       ~=Y)/length(Y)
% The error of testing set:
error_filtered_test  = 100*sum(classify(X_test_filtered, X_filtered, Y,'linear') ...,
                       ~=Y_test)/length(Y_test)

% d)



% % Plot KNN classifier for training data
% for i = 1:4
%     k = [1 5 25 151];
%     figure(i+2); plot(train_green(:,1), train_green(:,2),'go'); hold on;
%     plot(train_red(:,1), train_red(:,2),'ro');
%     title(sprintf('%d of Nearest Neighbor Classifier',k(i)));
%     % Knn classifier for training data (lacks error)
%     class_knn = knnclassify(sample, training, group, k(i)); % knn classifier
%     contour(linspace(min_x, max_x),linspace(min_y, max_y), reshape(class_knn, 100, 100), 1, 'b', 'linewidth', 2); %plot
%     axis([min_x max_x min_y max_y]); xlabel('x'); ylabel('y');
%     hold off;
% end