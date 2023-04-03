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

function q = expmap(p,v)
%Exponential map on the Riemannian manifold of Gaussian distributions
%where the Riemannian metric is the one inducing the 2-Wasserstein metric
%For more about this manifold, see

%Takatsu, Asuka. "Wasserstein geometry of Gaussian measures." Osaka Journal
%of Mathematics 48.4 (2011): 1005-1026.

%<p> is the basepoint of the exponential map and <v> is the initial speed
%in the tangent space.
p_mu = p{1};
p_K = p{2};

v_mu = v{1};
v_K = v{2};

n = numel(p_mu);
q_m = p_mu+v_mu;
q_K = (eye(n)+v_K)*p_K*(eye(n)+v_K);


q = {q_m,q_K};