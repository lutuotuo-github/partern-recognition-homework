function X_norm = data_central(X)

mu = mean_own(X);
X_norm = X - mu;

end