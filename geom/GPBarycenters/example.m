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

population = Vanavara_GPs;
%Visualize the data set of GPs
visualizePopulation(days,population,'b');
%Compute the mean
f_mean = Wbarycenter(population);
%Visualize the mean with oppacity 0.5
plotGP(days,f_mean,'r',0.5);
title('Population of GPs in blue, barycenter in red.')

%Plot geodesic between go GPs
f1 = population{1};
f2 = population{2};
v = logmap(f1,f2);
geodesic = @(t) expmap(f1,cellfun(@(x) t*x,v,'un',0));
M=10;
t = linspace(0,1,M);
figure
title('Geodesic between red and blue GPs. Click the image.')
plotGP(days,f1,'r',0.5);
plotGP(days,f2,'b',0.5);
for i=1:M
    %Compute the element on the principal geodesic at time t(i)
    gpi = geodesic(t(i));
    %Plot the GP
    [h1,h2]=plotGP(days,gpi,'g',0.5);
    w = waitforbuttonpress;
    %Hide the previous GP plot before plotting the next one
    set(h1,'Visible','off')
    set(h2,'Visible','off');
end