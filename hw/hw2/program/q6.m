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

% Since glmnet can not take Y is not from 1 to n where 1 to n must be
% contious, I remap it to 1-3 in this case.
Y_glmnet = Y;
for i=1:length(Y)
    if Y(i) == 3
      Y_glmnet(i) =1;
    end
    if Y(i) == 5
        Y_glmnet(i) = 2;
    end
    if Y(i) == 8
        Y_glmnet(i) = 3;
    end
end
Y_test_glmnet = Y_test;
for i=1:length(Y_test)
    if Y_test(i) == 3
      Y_test_glmnet(i) =1;
    end
    if Y_test(i) == 5
        Y_test_glmnet(i) = 2;
    end
    if Y_test(i) == 8
        Y_test_glmnet(i) = 3;
    end
end

coeff = glmnet(X_filtered, Y_glmnet, 'multinomial', glmnetSet); 
result = glmnetPredict(coeff, 'class', X_filtered); 
% The error of training set:
error_glmnet_train = 100*sum(result(:,length(coeff.lambda))  ...,
                       ~=Y_glmnet)/length(Y_glmnet)

result = glmnetPredict(coeff, 'class', X_test_filtered); 
% The error of testing set
error_glmnet_train = 100*sum(result(:,length(coeff.lambda))  ...,
                       ~=Y_test_glmnet)/length(Y_test_glmnet)

