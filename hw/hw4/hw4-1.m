% HW4 Q1
% Dong-Bang Tsai
clear;

N_sample = 4601;
Y=zeros(N_sample,1);
Y_type = -1*ones(N_sample,1); % 0 is training set, 1 is testing set.
X=zeros(N_sample,58);

scanformate = '';
for i=1:59
    scanformate = strcat(scanformate, ' %f');
end

fid1  = fopen('spam.data');
fid2 = fopen('spam.traintest');
tline1 = fgetl(fid1);
tline2 = fgetl(fid2);
for i=1:N_sample
    C = textscan(tline1,scanformate);
    for j=1:58
        X(i,j) = C{j};
    end
    Y(i) = C{58};
    C2 = textscan(tline2, '%f');
    Y_type = C2{1};
    tline1 = fgetl(fid1);
    tline2 = fgetl(fid2);
end
fclose(fid1);
fclose(fid2);

indx_train = find(Y_type == 0);
Y_train = Y(indx_train);
X_train = X(indx_train,:);

indx_test = find(Y_type == 1);
Y_test = Y(indx_test);
X_test = X(indx_test,:);

Y_train_est = classify(X_train,X_train, Y_train,'quadratic');
misclassified_rate_train_raw = sum(Y_train_est ~= Y_train)/length(Y_train);

Y_test_est = classify(X_test, X_train, Y_train,'quadratic');
misclassified_rate_test_raw = sum(Y_test_est ~= Y_test)/length(Y_test);

knots = [4 8 16 32 64];
for i = 1:5
    sp = spap2(knots(i),4,[1:256],X_train);
    SpData{i} = sp.coefs;
end

error_knots = zeros(5,1);
for i=1:5
    Y_sp = Y_train;
    X_sp = SpData{i};
   for j=1:10
       Y_sp_test =  Y_sp(334*(j-1) +1:334*j);
       Y_sp_train = Y_sp;
       Y_sp_train(334*(j-1) +1:334*j) =[];
       X_sp_test = X_sp(334*(j-1) +1:334*j,:);
       X_sp_train = X_sp;
       X_sp_train(334*(j-1) +1:334*j,:) = [];
       Y_sp_test_est = classify(X_sp_test,X_sp_train, Y_sp_train,'quadratic');
       error_knots(i) = error_knots(i) + sum(Y_sp_test_est ~= Y_sp_test)/length(Y_sp_test);
   end
end
error_knots = error_knots/10;

sp_test = spap2(32,4,[1:256],X_test);
Y_sp_train_est = classify(SpData{4},SpData{4}, Y_train,'quadratic');
misclassified_rate_train_sp = sum(Y_sp_train_est ~= Y_train)/length(Y_train);

Y_sp_test_est = classify(sp_test.coefs, SpData{4}, Y_train,'quadratic');
misclassified_rate_test_sp = sum(Y_sp_test_est ~= Y_test)/length(Y_test);
