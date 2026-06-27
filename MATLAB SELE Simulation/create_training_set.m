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


%% Emission-energy grid (used to integrate Sp(λ_emit, x) → SELE(x))
% calc_Sp2 scales the whole emission-band alpha by the random alpha_scale.
E_emit = 1240 ./ wavelength_PL;            % eV
[E_sorted, sort_idx] = sort(E_emit(:));    % ascending (required by trapz)

%% Dataset generation
n_samples = 1e4;
data = zeros(n_samples, x_res);

for i = 1:n_samples
    %% Semiconductor parameters
    p0 = 10^(log10(1e16) + (log10(1e19) - log10(1e16)) * rand); %1e19;
    
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

    %% Randomized parameters (log-uniform)
    D = 10^(log10(50) + (log10(200) - log10(50)) * rand);
    S = 10^(log10(1)  + (log10(1e7) - log10(1))  * rand);
    tau = 10^(log10(1e-10) + (log10(1e-6) - log10(1e-10)) * rand);

    % Emission-band absorption is hard to measure; vary it so the prior stays agnostic to it
    alpha_scale = 10^(log10(0.1) + (log10(10) - log10(0.1)) * rand);

    % Effective lifetime
    tau_eff = 1 / (1/tau + 1/tau_rad + 1/tau_auger);

    % Spectrally-resolved emission probability Sp(λ_emit, x)
    Sp_2d = calc_Sp2( ...
        x, wavelength_PL, p0, ni, tau_eff, S, D, ...
        n_k_wavelength, n_w_drude, k_w_drude, k_no_drude, ...
        alpha_scale);

    % Integrate over emission energy → true SELE(x)
    SELE = trapz(E_sorted, Sp_2d(sort_idx, :), 1);

    data(i, :) = SELE;
    figure()
    plot(SELE,YDataSource = 'data(i,:)');
    ylabel("data(i,:)");
    title("data(i,:)");
    legend("show");
    % pause;
    close all;
end

%% Save output
% save("sele_simulated_10k_curves_500_long.mat", "data");
writematrix(data, "sele_simulated_10k_curves_500_long.csv");