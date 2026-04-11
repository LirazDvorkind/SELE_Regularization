function [Sp] = calc_Sp2_single_wavelength(x, wavelength, p0, ni, tau_eff, S, D, n_k_wavelength, n, k_w_drude, k_no_drude, alpha_153)
% Calculates the photon emission probability Sp(x) for a single wavelength.
% Inputs:
%   x            - spatial points (cm), row vector of length M
%   wavelength   - scalar emission wavelength (nm)
%   p0           - equilibrium hole concentration (cm^-3)
%   ni           - intrinsic carrier concentration (cm^-3)
%   tau_eff      - effective lifetime (s)
%   S            - surface recombination velocity (cm/s)
%   D            - diffusion coefficient (cm^2/s)
%   n_k_wavelength, n, k_w_drude, k_no_drude - optical constants
%   alpha_153    - absorption coefficient override at this wavelength (cm^-1)
%
% Output:
%   Sp           - row vector of length M (same size as x)

%% Physical constants
Kb   = 8.6173e-5;          % eV/K
hbar = 6.582119569e-16;    % eV·s
TK   = 299;
c0   = 2.998e10;           % cm/s

%% Diffusion length
Ln = sqrt(D * tau_eff);

%% Optical parameters (scalar — single wavelength)
n_interp   = interp1(n_k_wavelength, n,          wavelength);
k_interp_b = interp1(n_k_wavelength, k_no_drude, wavelength);

alpha     = alpha_153;                          % use the provided alpha directly
alpha_b   = 4*pi * k_interp_b / (wavelength * 1e-7);

theta_c = asin(1 / n_interp);
[~, T, ~] = Fresnel(n_interp, 1, 0);

%% K prefactor (scalar)
E  = 1240 / wavelength;
B  = 1/(pi^2 * hbar^3 * c0^2 * ni^2) * alpha_b * n_interp^2 * E^2 * exp(-E / (Kb * TK));
K2 = T * B * p0 * tau_eff * (1 - cos(theta_c)) / 4;
K1 = K2 / (1 - (alpha * Ln)^2);

%% SELE profile (vector over x)
A0 = (D - S*Ln) / (S*Ln + D);
A1 = 2;
A2 = A0 - 1 - Ln * alpha * (A0 + 1);

Sp = K1 * (A1 * exp(-alpha * x) + A2 * exp(-x / Ln));

end