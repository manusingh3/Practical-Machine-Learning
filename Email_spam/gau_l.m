function [error] = gau_l(data,labels)

linclass = fitcdiscr(data,labels);
pred_lin = predict(linclass,data);

error = mean(sign(pred_lin)~=labels);

end

