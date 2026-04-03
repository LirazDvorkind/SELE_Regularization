function [Sp] = calc_Sp2(x,wavelength,p0,ni,tau_eff,S, D,n_k_wavelength,n,k_w_drude,k_no_drude, alpha_153);
% This function claculates the photon energy dependent probability to emit
% a photon after a photogeneration event at point x. The inputs are:
%  x: the points in space where Sp is calculated (cm)
% wavelength: the wavelegnth of the emitted photon (wavelength=1240/E where
% E is the photon energy)
% p0: The equilibrium hole concentration (cm^-3, this is a known constant it is a
% part of K2 from the file)
% ni- the intrinsinc carrier concentration (cm^-3, this is a known constant it is a
% part of K2 from the file)
% tue_eff- the effective lifetime (s). This is unkown and should be
% extracted (it makes K2 so if K2 is extracted, so is tau_eff)
% S- the surface recombination velocity (cm/s). This is unkown and should be
% extracted
% D- the diffusion coefficient (cm^2/s). This is unkown and should be
% extracted
% n_k_wavelength,n,k_w_drude,k_no_drude- The optical constants of the
% !!! notice that \alpha from the document is calculated here from the
%optial constants. Instead of the calculation here, we would like to find alpha as a part of the learning process.  
% material. Those are known.
%% Physical contants
Kb=8.6173e-5; % Boltzmans Constant in eV/K
hbar=6.582119569e-16; % the reduced Planks constant (eVs)
TK=299;
c0=2.998e10; % Speed of light in vaccum (cm/s)

x_matrix=repmat(x,length(wavelength),1);

Ln=sqrt(D*tau_eff);

%% Optical parameters
n_interp=interp1(n_k_wavelength,n,wavelength);
k_interp=interp1(n_k_wavelength,k_w_drude,wavelength);
alpha=4*pi.*k_interp./(wavelength*1e-7);  %in 1/cm
alpha(153)=alpha_153;
alpha_matrix=repmat(alpha',1,length(x));
theta_c=asin(1./n_interp);
[ ~,T,~ ] = Fresnel( n_interp,1,0 );

k_interp=interp1(n_k_wavelength,k_no_drude,wavelength);
alpha_b=4*pi.*k_interp./(wavelength*1e-7);  %in 1/cm

%% Calculating the K prefactor

E=1240./wavelength;
B=1/(pi^2*hbar^3*c0^2*ni^2).*alpha_b.*n_interp.^2.*E.^2.*exp(-E/(Kb*TK));
K2=T.*B.*p0*tau_eff.*(1-cos(theta_c))./4;
K1=K2./(1-(alpha*Ln).^2);
K_matrix=repmat(K1',1,length(x));

A0=(D-S*Ln)/(S*Ln+D);
A1=2;
A2=A0-1-Ln*alpha_matrix.*(A0+1);

Sp=K_matrix.*(A1.*exp(-alpha_matrix.*x_matrix)+A2.*exp(-x_matrix/Ln));

end

