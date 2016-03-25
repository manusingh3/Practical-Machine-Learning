function [error] = avgp(data,labels,testdata,testlabels)

w = zeros(64,57); b=0;
pred= zeros(1,size(data,1));
error = 0;
%u = zeros(3065,57); beta = 0;
%c=1;

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
    pred = weight * (testdata)';

error = mean(sign(pred')~=testlabels);
end


            
            
        


