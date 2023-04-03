% Copyright (C) 2017 Anton Mallasto
% 
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
%
% Author: Anton Mallasto <mallasto at di dot ku dot dk>

function d = W(f1,f2)
%Compute the 2-Wasserstein distance between Gaussian distributions
%f1=(m1,K1) and f2=(m2,K2). f1,f2 are given in cell format, with first
%element being the mean vector, and the second element is the covariance
%matrix.

%%Load mean vectors and covariance matrices.
%%
m1 = f1{1}; m2 = f2{1};
K1 = f1{2}; K2 = f2{2};

%%Compute the distance
%%
covDist =trace(K1)+trace(K2)-2*trace(sqrtm(sqrtm(K1)*K2*sqrtm(K1)));
l2norm =norm(m1-m2,'fro')^2;
d = real(sqrt(covDist+l2norm));
