% HW3 Q2, ESL 18.9
% Dong-Bang Tsai
clear;
N = 100; 
p = 200;
X = randn(N, p);
Y = randn(N,1);
for i=1:N
   if Y(i) > 0
       Y(i) = 1;
   else
       Y(i) = -1;
   end
end
[U,S,V] = svd(X, 'econ');
beta_p = (V/S)*U'*Y;
% The following is the projection distance obtained by ESL18.8 (c)
D_p = X*beta_p/sqrt(sum(beta_p.^2));

SVMStruct = svmtrain(X,Y);
Group = svmclassify(SVMStruct,X);
beta_s = SVMStruct.SupportVectors'*SVMStruct.Alpha;
% The following is the projection distance obtained by SVM
D_s = -X*beta_s/sqrt(sum(beta_s.^2));

x_indx = [1:N];
plot(x_indx, D_p, 'bo', x_indx, D_s, 'ro');
xlabel('index of Data'); ylabel('Projections');
legend('ESL18.8 (c)','SVM')