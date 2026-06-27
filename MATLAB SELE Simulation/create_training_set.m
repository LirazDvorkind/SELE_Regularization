%% Initialization
clear; close all; clc;
rng(12); % Reproducibility

%% Load data
load Incident_wavelength_dependent_PL % wavelength_PL
load optical_constnats_w_wo_Drude.mat % optical constants

%% Spatial domain
W = 30e-4; % cm
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

%% Semiconductor parameters
NC = 8.63e13 * TK^(3/2);
NV = 1.83e15 * TK^(3/2);

%% Recombination parameters
C_auger = 15e-30;
B0      = 2.5e-10;

%% Dataset generation
n_samples = 1e2;
data = zeros(n_samples, x_res);

% Output file name - can be .mat or .csv, the code will save in the correct
% format per the extension
data_file_output_name = sprintf("sele_simulated_%d_curves_%d_long.csv", ...
    n_samples, x_res);
is_debug = true; % Place breakpoint on the "close all" and hit Continue

for i = 1:n_samples
    % Randomized parameters
    % 1. p0 randomization
    % Semiconductor parameters
    p0 = 10^(log10(1e16) + (log10(1e19) - log10(1e16)) * rand); %1e19; 
    
    dEg = ( ...
        9.71  * (p0/1e18)^(1/3) + ...
        12.19 * (p0/1e18)^(1/4) + ...
        3.88  * (p0/1e18)^(1/2) ) / 1000;
    
    Eg = 1.519 ...
       - 5.405e-4 * TK^2 / (TK + 204) ...
       - 0.8 * dEg;
    
    ni = sqrt(NC * NV) * exp(-Eg / (2 * Kb * TK));
    
    % Recombination parameters    
    tau_auger = 1 / (p0^2 * C_auger);
    tau_rad   = 1 / (p0 * B0);
    
    % 2. D, SRV, tau, emission absorption coeff randomization:
    D = 50 + (200 - 50) * rand;
    S = 10^(log10(200)  + (log10(1e7) - log10(200))  * rand);
    tau = 10^(log10(5e-9) + (log10(2.5e-7) - log10(5e-9)) * rand);

    % Emission absorption is hard to measure so vary it so the prior stays agnostic to it
    alpha_scale = 10^(log10(0.1) + (log10(10) - log10(0.1)) * rand);

    % Effective lifetime
    tau_eff = 1 / (1/tau + 1/tau_rad + 1/tau_auger);
    
    % At lower doping the FCA drops
    k_eff = k_no_drude + (k_w_drude - k_no_drude) * (p0 / 1e19);
    
    % Spectrally-resolved emission probability Sp(λ_emit, x)
    %  If a photon is absorbed at depth x, what is the probability it 
    %  produces an observable PL photon at emission wavelength λ_emit?
    Sp_2d = calc_Sp2( ...
        x, wavelength_PL, p0, ni, tau_eff, S, D, ...
        n_k_wavelength, n_w_drude, k_eff, k_no_drude, ...
        alpha_scale);

    % Integrate over emission energy → true SELE(x)
    SELE = trapz(E_sorted, Sp_2d(sort_idx, :), 1);
    
    data(i, :) = SELE;

    if mod(i, round(n_samples * 0.05)) == 0
        fprintf('Generated %d / %d samples (%.0f%%)\n', i, n_samples, 100*i/n_samples);
    end

    %% Debug plots
    if is_debug
        figure()
        plot(SELE,YDataSource = 'data(i,:)');
        ylabel("data(i,:)");
        title("data(i,:)");
        legend("show");
        close all;
    end
end

%% Save output
[~,~,ext] = fileparts(data_file_output_name);
if ext == ".mat"
    save(data_file_output_name, "data");
else
    % Save as .csv
    writematrix(data, data_file_output_name);
end
