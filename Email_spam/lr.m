
function [error] = lr(data,labels,testdata,testlabels)

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

 [failedPred] = glmval(logitCoef,testdata,'logit');
 
 outlabel = zeros(size(testdata,1),1);
 for i = 1:size(testdata,1)
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
