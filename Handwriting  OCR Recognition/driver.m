error = zeros(10,4);
for i = 1:10
    [out,error(i,1)] = knne(data,labels,testdata,testlabels,1000);
end

for i = 1:10
    [out,error(i,2)] = knne(data,labels,testdata,testlabels,2000);
end
for i = 1:10
    [out,error(i,3)] = knne(data,labels,testdata,testlabels,4000);
end
for i = 2:10
    [out,error(i,4)] = knne(data,labels,testdata,testlabels,8000);
end


x = [1000 2000 4000 8000] ;
y= mean(error);
e = std(y) * ones(size(x));
figure
errorbar(x,y,e)

