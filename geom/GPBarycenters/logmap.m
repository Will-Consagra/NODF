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

function [v] = logmap(p,q)
%The logarithmic map from GD <p> to <q> on the Riemannian manifold of 
%GDs with the Wasserstein metric. For more about this manifold, see

%Takatsu, Asuka. "Wasserstein geometry of Gaussian measures." Osaka Journal
%of Mathematics 48.4 (2011): 1005-1026.

%Note that the log map has two components, as the manifold is a product of
%L2 mean vectors and symmetric positive definite matrices
n = numel(p{1});
v_mu = q{1}-p{1};
v_K = transportMap(p{2},q{2})-eye(n);
v = {v_mu,v_K};
