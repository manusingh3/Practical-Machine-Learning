%Function definition of Model 1
function [error] = avgp(data,labels)

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
    pred = weight * (data)';

error = mean(sign(pred')~=labels);
end

%Function definition of Model 2

function [error] = lr(data,labels)

ynew= zeros(length(data),1);
error = 0;
e=0;

for i=1:size(data,1)
    if labels(i) == -1
        ynew(i)=0;
            
    else
     ynew(i)=1;
    end
end 

 [logitCoef] = glmfit(data,ynew,'binomial','logit');

 [failedPred] = glmval(logitCoef,data,'logit');
 
 outlabel = zeros(size(data,1),1);
 for i = 1:size(data,1)
     if failedPred(i) >0.5
         outlabel(i) = 1;
         
     else 
         outlabel(i) =-1;
     end
   if outlabel(i)~= labels(i)
         e = e+1;
      end   
 end
 error = e/size(data,1);

end
%Function definition of Model 3
function [error] = gau_l(data,labels)

linclass = fitcdiscr(data,labels);
pred_lin = predict(linclass,data);

error = mean(sign(pred_lin)~=labels);

end
%Function definition of Model 4
function [error] = gau_q(data,labels)

quadclass = fitcdiscr(data,labels,...
    'discrimType','quadratic');

pred_quad = predict(quadclass,data);
error = mean(sign(pred_quad)~=labels);
end
%Function definition of Model 5
function [error] = expander_p(data,labels)
c=114;
for i = 1:57
    for j = i+1:57
        c=c+1;
        data(:,c)= data(:,i).*data(:,j);
    end
end
w = zeros(64,1710); b=0;
pred= zeros(1,size(data,1));
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
    pred = weight * (data)';

error = mean(sign(pred')~=labels);
end
%Function definition of Model 6
function [error] = expander_l(data,labels)
c=114;
for i = 1:57
    for j = i+1:57
        
       data(:,c)= data(:,i).*data(:,j);
       c=c+1;
    end
end


ynew= zeros(size(data,1),1);

e=0;
size(data,1)
for i=1:3065
    if labels(i) == -1
        ynew(i)=0;
            
    else
     ynew(i)=1;
    end
end 

 [logitCoef] = glmfit(data,ynew,'binomial','logit');
 %logitFit = glmval(logitCoef,weight,'logit');
 [failedPred] = glmval(logitCoef,data,'logit');
 
 outlabel = zeros(size(data,1),1);
 for i = 1:size(data,1)
     if failedPred(i) >0.5
         outlabel(i) = 1;
         
     else 
         outlabel(i) =-1;
     end
   if outlabel(i)~= labels(i)
         e = e+1;
      end   
 end
 error = e/size(data,1);
end



error =zeros(6,1)

error(1)=avgp(data,labels);
error(2)=lr(data,labels);
error(3)=gau_l(data,labels);
error(4)=gau_q(data,labels);
error(5)=expander_p(data,labels);
error(6)= expander_l(data,labels);
[~,I]=min(error);

fncs = ['avgp','lr','gau_l','gau_q','expander_p','expander_l'];
feval(fncs(I),testdata,testlabels)
