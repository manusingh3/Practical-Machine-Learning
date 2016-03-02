function [out_label, error] = knne(data,labels,testdata,testlabels,n)
sel = randsample(60000,n);
datanew = data(sel,:);
labelnew = labels(sel);
dist = zeros(n,1);
out_label = zeros(1000,1);
error = 0;
for m = 1:1000
    for k = 1:n
        dist(k) = (testdata(m,:)-datanew(k,:))*((testdata(m,:)-datanew(k,:))');
    end
    [~,I]=min(dist);
    
    %disp(labelnew(I));
    out_label(m) = labelnew(I);
    
end
 for e = 1:1000
     if out_label(e)~= testlabels(e)
         error = error+1;
     end
 end
         
     
 end