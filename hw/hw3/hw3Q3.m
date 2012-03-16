% HW3 Q3, ESL 18.9
% Dong-Bang Tsai
clear;

% type of phonemes, 1)aa  2)ao 3)dcl 4)iy 5)sh
N_sample = 4509;
Y=zeros(N_sample,1);
Y_type = -1*ones(N_sample,1); % 0 is training set, 1 is testing set.
X=zeros(N_sample,256);

% This part of reading data takes most of mine time.
scanformate = '%d';
for i=1:256
    scanformate = strcat(scanformate, ', %f');
end
scanformate = strcat(scanformate, ', %s');
fid = fopen('phoneme.data');
tline = fgetl(fid);
tline = fgetl(fid);
for i=1:N_sample
    C = textscan(tline,scanformate);
    for j=1:256
        X(i,j) = C{j+1};
    end
    [tok,rem] = strtok(C{258},',');
    if strcmp(tok, 'aa')
        Y(i) = 1;
    elseif  strcmp(tok, 'ao')
        Y(i) = 2;
    elseif strcmp(tok,'dcl')
        Y(i) = 3;
    elseif strcmp(tok,'iy')
        Y(i) = 4;
    elseif strcmp(tok,'sh')
        Y(i) = 5;
    end
    [tok,rem] = strtok(rem,',');
    [tok,rem] = strtok(tok,'.');
    if strcmp(tok,'train')
        Y_type(i) = 0;
    elseif strcmp(tok,'test')
        Y_type(i) = 1;
    end
    tline = fgetl(fid);
end
fclose(fid);

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
