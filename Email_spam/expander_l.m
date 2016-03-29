function [error] = expander_l(data,labels,testdata,testlabels)
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
 [failedPred] = glmval(logitCoef,testdata,'logit');
 
 outlabel = zeros(size(testdata,1),1);
 for i = 1:size(data,1)
     if failedPred(i) >0.5
         outlabel(i) = 1;
         
     else 
         outlabel(i) =-1;
     end
   if outlabel(i)~= testlabels(i)
         e = e+1;
      end   
 end
 error = e/size(testdata,1);
end
