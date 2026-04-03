function [ R,T,theta_out ] = Fresnel( ni,nt,theta_in )
%This function calculates the Reflected and transmitted POWER across the
%interface between i (incident) and t (target). The light is assumed to be
%non prolerized
theta_out=asin(ni./nt.*sin(theta_in));

Rs=((ni.*cos(theta_in)-nt.*cos(theta_out))./(ni.*cos(theta_in)+nt.*cos(theta_out))).^2;
Rp=((ni.*cos(theta_out)-nt.*cos(theta_in))./(ni.*cos(theta_out)+nt.*cos(theta_in))).^2;

R=(Rs+Rp)/2;
R(imag(R)~=0)=1;
T=1-R;

end

