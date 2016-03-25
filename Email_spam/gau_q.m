function [error] = gau_q(data,labels)

quadclass = fitcdiscr(data,labels,...
    'discrimType','quadratic');

pred_quad = predict(quadclass,data);
error = mean(sign(pred_quad)~=labels);
end