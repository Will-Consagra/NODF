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

function [h1,h2] = plotGP(xs,f,c,alpha)
%Visualize GP <f> with domain <xs> in color <c> with transparency <alpha>.
if nargin<3
    alpha=0.1;
    c='k';
end
mu = f{1}'; K = f{2};
s2 = diag(K)';
s = sqrt(s2);
envelope = [mu+s flip(mu-s,2)];

hold on;
h1=fill([xs flip(xs,2)], envelope,c,'EdgeAlpha',alpha,'FaceAlpha',alpha,...
    'EdgeColor',c);
h2=plot(xs, mu,'color',c,'LineWidth',1);
set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

