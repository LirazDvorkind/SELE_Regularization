%% Initialization
clear; close all; clc;
rng(12);   % Reproducibility

%% Load data
load Incident_wavelength_dependent_PL              % wavelength_PL
load optical_constnats_w_wo_Drude.mat               % optical constants

%% Spatial domain
W = 30e-4;                 % cm
x_res = 500;
x = linspace(0, W, x_res);

%% Physical constants
Kb   = 8.6173e-5;          % eV/K
hbar = 6.582119569e-16;    % eV·s
TK   = 299;                % K
c0   = 2.998e10;           % cm/s

%% Semiconductor parameters
p0 = 1e19;

NC = 8.63e13 * TK^(3/2);
NV = 1.83e15 * TK^(3/2);

dEg = ( ...
    9.71  * (p0/1e18)^(1/3) + ...
    12.19 * (p0/1e18)^(1/4) + ...
    3.88  * (p0/1e18)^(1/2) ) / 1000;

Eg = 1.519 ...
   - 5.405e-4 * TK^2 / (TK + 204) ...
   - 0.8 * dEg;

ni = sqrt(NC * NV) * exp(-Eg / (2 * Kb * TK));

%% Recombination parameters
C_auger = 15e-30;
B0      = 2.5e-10;

tau_auger = 1 / (p0^2 * C_auger);
tau_rad   = 1 / (p0 * B0);

%% Optical parameters
alpha_idx = 153;   % wavelength index with learnable absorption

%% Dataset generation
% Stratified sampling: 30% of curves forced to have surface dip (A0 < 0).
% Dip condition: S * Ln > D  =>  S > sqrt(D / tau_eff).
% Achieved by sampling high-S / low-D stratum for the dip fraction.
n_samples     = 1e3;
dip_fraction  = 0.30;
n_dip         = round(n_samples * dip_fraction);   % ~30k dip curves
n_nodip       = n_samples - n_dip;                 % ~70k regular curves

data = zeros(n_samples, x_res);

for i = 1:n_samples

    if i <= n_nodip
        % --- Regular stratum: full wide parameter ranges ---
        D     = 10^(log10(5)    + (log10(500)  - log10(5))    * rand);
        S     = 10^(log10(1e2)  + (log10(1e9)  - log10(1e2))  * rand);
        tau   = 10^(log10(1e-10)+ (log10(1e-7) - log10(1e-10))* rand);
    else
        % --- Dip stratum: high S and low D to ensure A0 < 0 ---
        % S in [1e5, 1e9],  D in [5, 100]
        D     = 10^(log10(5)    + (log10(100)  - log10(5))    * rand);
        S     = 10^(log10(1e5)  + (log10(1e9)  - log10(1e5))  * rand);
        tau   = 10^(log10(1e-10)+ (log10(1e-7) - log10(1e-10))* rand);
    end

    alpha_153 = 150 + (15000 - 150) * rand;

    % Effective lifetime
    tau_eff = 1 / (1/tau + 1/tau_rad + 1/tau_auger);

    % Photon emission probability (single wavelength only)
    Sp = calc_Sp2_single_wavelength( ...
        x, wavelength_PL(alpha_idx), p0, ni, tau_eff, S, D, ...
        n_k_wavelength, n_w_drude, k_w_drude, k_no_drude, ...
        alpha_153);

    data(i, :) = Sp;
end

%% Save output
% save("sele_simulated_100k_curves_500_long.mat", "data");
writematrix(data, "sele_simulated_1000_curves_500_long_more_dip.csv");
