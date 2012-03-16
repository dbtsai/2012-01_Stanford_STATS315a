function [X, X2] = preprocessing(X,X2)

[raw col] = size(X);

mean_val = zeros(col,1);
for i=1:col
    mean_val(i) = mean(X(:,i)); 
    for j=1:raw
        if X(j,i) > mean_val(i)
            X(j,i) = 1;
        else
            X(j,i) = 0;
        end
    end
end

[raw col] = size(X2);

for i=1:col
    for j=1:raw
        if X2(j,i) > mean_val(i)
            X2(j,i) = 1;
        else
            X2(j,i) = 0;
        end
    end
end
