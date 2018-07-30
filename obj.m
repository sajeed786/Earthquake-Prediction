function [lb,ub,dim,fobj] = obj()
lb=-1;
ub=1;
dim=40;
fobj = @get_cost_mse;
end

function mse = get_cost_mse(y,x,w)
%fprintf('inside cost_mse function ');
pred_random = x * w';
O = ones(437,1);
act_op = O ./ (O + exp(-pred_random));
maxm = max(y);
minm = min(y);
denorm_op = ((act_op - 0.1) / 0.8) * (maxm - minm) + minm;
error = y - denorm_op;
error = error .^ 2;
mse = mean(error);
end