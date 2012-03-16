function Ytest = NaiveBayesclassify(Xtrain,Ytrain, Xtest)

[row col] = size(Xtrain);
n_class = max(Ytrain)+1;

psi_y = zeros(n_class,1);

for i=1:n_class
    psi{i}    = zeros(col,1);
    psi_y(i) = length( find(Ytrain==(i-1) ))/length(Ytrain);
    indx_y{i} = find(Ytrain==(i-1));   
    n_y{i}    = length(indx_y{i});
end


for i=1:col
    for j=1:n_class
        result = Xtrain(indx_y{j},:);    
        psi{j}(i) = (length(find( result(:,i)==1)) + 1)/( n_y{j} + 2 );
    end    
end

[row col] = size(Xtest);
Ytest = zeros(row,1);
for i=1:row
    class_score = zeros(n_class,1);
    for j=1:n_class
        for k=1:col
            if Xtest(i,k)  == 1
                logP = log( psi{j}(k) );
            else
                logP = log( 1 - psi{j}(k) );
            end
            class_score(j) = class_score(j) + logP;
        end
        class_score(j) = class_score(j) + log(psi_y(j));
    end
    [val, indx] = max(class_score);
    Ytest(i) = indx -1;
end

end