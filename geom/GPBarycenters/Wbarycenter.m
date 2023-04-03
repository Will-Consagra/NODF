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

function f_mean = Wbarycenter(population,lambda,err)
%Compute the Wasserstein barycenter of a population of GPs.
%<population> is a cell with m Gaussian distributions given by cells
%consisting of a mean vector and a covariance matrix.
%<lambda> is a 1xm vector consisting of the barycentric weights.
%<err> is the error margin.


%%Initialization
%%
m = numel(population); %Amount of Gaussians
for i=1:m
    fi = population{i};
    mi = fi{1};
    Ki = fi{2};
    means(:,i) = mi;
    Covmats(:,:,i) = Ki;
end
d = size(means, 1); %Dimension of the Gaussians

%Constant limiting the amount of iterations
uplimit = 10^2;

%If error margin is not specified, it is set to 1e-6.
if nargin < 3
    err = 1e-8;
    %If weights are not specified, uniform weights are chosen.
    if nargin < 2
        lambda = (1/m)*ones(1,m);
    end
end
%%Iteration
%%
%The barycenter is the fixed point of the map F. We solve for the fixed
%point by the following iteration, that is guaranteed to converge in this
%case (REFERENCE).

K = Covmats(:,:,1);
K_next = F(K,Covmats,lambda);
count = 0;
while(W({0,K},{0,K_next})>err && count < uplimit)
    K=K_next;
    K_next = F(K,Covmats,lambda);
    count=count+1;
end

if count==uplimit
    display('Barycenter did not converge.')
end
mu_mean = sum(repmat(lambda, d, 1).*means,2);
f_mean = {mu_mean, K_next};
end
%----------------------------------------------------------------------------

%The covariance matrix of the barycenter is the fixed point of the
%following map F.
function T = F(K,Covmats,lambda)
sqrtK = sqrtm(K);
d = size(Covmats,1);
m = numel(lambda);
T=zeros(d,d);
for i=1:m
    T= T + lambda(i)*sqrtm(sqrtK*Covmats(:,:,i)*sqrtK);
end
T = sqrtK\(T^2/sqrtK);
end

function E = energy(f,population,lambda)
E = 0;
for i=1:numel(population)
    E = E+lambda(i)*W(f,centerGP(population{i}))^2;
end
end



