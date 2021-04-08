function varx = var_own(x)

mu = mean_own(x);
[a b] = size(x);
varx = sum((x-mu)'*(x-mu))/(a-1);

end