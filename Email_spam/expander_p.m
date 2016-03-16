function [error] = expander_p(data,labels,testdata,testlabels)
c=114;
for i = 1:57
    for j = i+1:57
        c=c+1;
        data(:,c)= data(:,i).*data(:,j);
    end
end
w = zeros(64,1710); b=0;
pred= zeros(1,size(testdata,1));
error = 0;

for i=1:64
    
    for j=(1:size(data,1))
        
        a = (w(i,:)*data(j,:)')*(labels(j));
        if a <= 0
            w(i,:) = w(i,:) + labels(j)*(data(j,:));
            
             else
            w(i,:) = w(i,:);
            if i<64
                w(i+1,:) = w(i,:);
            end
        end 
        
    end
end
    weight = mean(w);
    size(weight)
    pred = weight * (testdata)';

error = mean(sign(pred')~=testlabels);
end
