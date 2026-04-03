clear all
close all
clc

load Incident_wavelength_dependent_PL
wavelength_PL_res=length(wavelength_PL);
load optical_constnats_w_wo_Drude.mat
%% Need to convert everything to cm !!! %%
W=30e-4;%300*1e-4; %in cm
x_res=100;%1001;
x=linspace(0,W,x_res);
x_matrix=repmat(x,wavelength_PL_res,1);

inicdent_wavelength=wavelength_in;


% Ln=2000;


%% Physical contants
Kb=8.6173e-5; % Boltzmans Constant in eV/K
hbar=6.582119569e-16; % the reduced Planks constant (eVs)
TK=299;
c0=2.998e10; % Speed of light in vaccum (cm/s)

% Semiconductor parmeters
p0=1e19; %from paper

NC=8.63e13*TK^(1.5); %Conduction band density of states (1/cm^3)
NV=1.83e15*TK^(1.5); %Valance band density of states (1/cm^3)
dEg=(9.71*(p0/1e18)^(1/3)+12.19*(p0/1e18)^(1/4)+3.88*(p0/1e18)^(1/2))/1000; %Band gap Narrowing according to Jaint and Roulston

Eg=1.519-5.405e-4*TK^2/(TK+204)-dEg*0.8;  %Band gap
ni=sqrt(NV*NC)*exp(-Eg/(2*Kb*TK));  %


%% Matrial properties (all taken from the paper)

% Recombination paramters
%the values from the paper
% C_aguer=15e-30;
% B0=2.5e-10; % (cm^6/s) from paper
% tau=0.6e-9/10; % The SRH lifetime Ln=sqrt(Dtau)-->D=Ln^2/tau 
% D=0.026*3000; %in cm^2/s
% S=4e7*10; % in cm/s

C_aguer=15e-30;
B0=2.5e-10; % (cm^6/s) from paper
tau=0.6e-9; % The SRH lifetime Ln=sqrt(Dtau)-->D=Ln^2/tau 
D=0.026*3000; %in cm^2/s
S=4e7*10; % in cm/s

tau_auger=1/(p0^2*C_aguer);
tau_rad=1/(p0*B0);

tau_eff=(1/tau+1/tau_rad+1/tau_auger)^-1;
Ln=sqrt(D*tau_eff);


%% Optical parameters
n_interp=interp1(n_k_wavelength,n_w_drude,wavelength_PL);
k_interp=interp1(n_k_wavelength,k_w_drude,wavelength_PL);
k_interp_no_drude=interp1(n_k_wavelength,k_no_drude,wavelength_PL);

alpha=4*pi.*k_interp./(wavelength_PL*1e-7);  %in 1/cm 
alpha_matrix=repmat(alpha',1,x_res);
theta_c=asin(1./n_interp);
[ ~,T,~ ] = Fresnel( n_interp,1,0 );

k_interp=interp1(n_k_wavelength,k_no_drude,wavelength_PL);
alpha_b=4*pi.*k_interp_no_drude./(wavelength_PL*1e-7);  %in 1/cm 

Sp = calc_Sp2(x,wavelength_PL,p0,ni,tau_eff,S, D,n_k_wavelength,n_w_drude,k_w_drude,k_no_drude);
E=1240./wavelength_PL;



figure
s=surf(x*1e4,E,Sp);
s.LineStyle='none';
xlabel('x (\mum)')
ylabel('Energy (eV)')
zlabel('S_p (1/eV)')
% 
% 
% 
% figure
% m=mesh(x*1e4,E,Sp);
% % m.LineStyle='none';
% xlabel('x (\mum)')
% ylabel('Energy (eV)')
% zlabel('S_p (1/eV)')

figure
colormap(parula)
hold on
contour(E,x*1e4,Sp','Fill','on','LevelList',linspace(0,max(max(Sp)),100))
contour(E,x*1e4,Sp','Fill','off','LevelList',linspace(0,max(max(Sp)),10),'LineColor','k','ShowText','on')
cb=colorbar;
set(gca,'FontSize',14)
xlabel('Energy (eV)')
ylabel('Depth (\mum)')
title('S_p (1/eV)')
ylim([0 10]);
