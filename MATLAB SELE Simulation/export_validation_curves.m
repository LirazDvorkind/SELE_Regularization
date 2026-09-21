%% Reference curves for validating the Python port of the simulator.
% Writes validation_reference.mat next to this script, holding the drawn parameters, the
% MATLAB SELE curves, the depth grid and the optical constants.
% src/forward_model/standalones/validate_matlab_port.py reads it and checks simulate_sele()
% reproduces the curves.
%
% The setup below mirrors create_training_set.m exactly; compute_sele_curve.m is that
% script's inner loop factored out, so it is the right reference to compare against.
clear; close all; clc;
rng(2024);

%% Load data
load Incident_wavelength_dependent_PL % wavelength_PL
load optical_constnats_w_wo_Drude.mat % optical constants

%% Spatial domain
W = 30e-4; % cm
x_res = 500;
x = linspace(0, W, x_res);

%% Physical constants
Kb   = 8.6173e-5;          % eV/K
hbar = 6.582119569e-16;    % eV*s
TK   = 299;                % K
c0   = 2.998e10;           % cm/s

%% Emission-energy grid
E_emit = 1240 ./ wavelength_PL;
[E_sorted, sort_idx] = sort(E_emit(:));

%% Semiconductor parameters
NC = 8.63e13 * TK^(3/2);
NV = 1.83e15 * TK^(3/2);

%% Recombination parameters
C_auger = 15e-30;
B0      = 2.5e-10;

%% Parameter draws, same laws as create_training_set.m
n_validation = 200;
params = zeros(n_validation, 5);
SELE_all = zeros(n_validation, x_res);

for i = 1:n_validation
    p0          = 10^(log10(1e16)  + (log10(1e19)   - log10(1e16))  * rand);
    % Log-uniform from 5, matching src/forward_model/parameters.py rather than
    % create_training_set.m, so the check covers the short diffusion lengths the
    % widened box reaches and the 1-(alpha*Ln)^2 pole they sit near.
    D           = 10^(log10(5)     + (log10(200)    - log10(5))     * rand);
    S           = 10^(log10(200)   + (log10(1e7)    - log10(200))   * rand);
    tau         = 10^(log10(5e-9)  + (log10(2.5e-7) - log10(5e-9))  * rand);
    alpha_scale = 10^(log10(0.1)   + (log10(10)     - log10(0.1))   * rand);

    params(i, :) = [p0, D, S, tau, alpha_scale];

    SELE_all(i, :) = compute_sele_curve(p0, D, S, tau, alpha_scale, ...
        x, wavelength_PL, E_sorted, sort_idx, NC, NV, Kb, TK, C_auger, B0, ...
        n_k_wavelength, n_w_drude, k_w_drude, k_no_drude);
end

%% Also dump the optical constants, so the port's .fig-derived copy can be checked
optical = [n_k_wavelength(:), n_w_drude(:), n_no_drude(:), k_w_drude(:), k_no_drude(:)];

%% Save as .mat: full double precision, no text round-trip to muddy the comparison
here = fileparts(mfilename('fullpath'));
save(fullfile(here, 'validation_reference.mat'), 'params', 'SELE_all', 'optical', 'x');

fprintf('Wrote %d validation curves (%d depth points).\n', n_validation, x_res);
