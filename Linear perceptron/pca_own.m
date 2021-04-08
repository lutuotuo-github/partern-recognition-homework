function [res, U_1] = pca_own(obj, dimen)

obj_norm = data_central(obj);
[U_1, S_1] = eig((1/size(obj_norm,1)) * obj_norm' * obj_norm);
res = projectData(obj, U_1, dimen);

end