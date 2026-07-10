%% Parameter Sensitivity Analysis for SELE Reconstruction
% This script analyzes how each tunable parameter affects the SELE profile.
% Five parameters are systematically varied while others are held fixed.
%
% PARAMETERS (with typical ranges from training data generation):
%   1. p0 (hole doping):              1e16   to 1e19   cm^-3    [LOGSPACE]
%   2. D (diffusion coefficient):     50     to 200    cm^2/s   [LINEAR]
%   3. S (surface recomb velocity):   200    to 1e7    cm/s     [LOGSPACE]
%   4. tau (effective lifetime):      5e-9   to 2.5e-7 s        [LOGSPACE]
%   5. alpha_scale (absorption):      0.1    to 10     -        [LOGSPACE]
%
% PHYSICS:
%   - p0: Controls bandgap (Eg), intrinsic carrier conc (ni), radiative/Auger lifetimes
%   - D: Diffusion length Ln = sqrt(D*tau) → affects minority carrier transport
%   - S: Surface recombination velocity → controls surface SELE dip & diffusion length
%   - tau: Effective lifetime → controls overall SELE magnitude & diffusion length
%   - alpha_scale: Scales emission-band absorption → affects spectral response

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

%% Emission-energy grid
E_emit = 1240 ./ wavelength_PL;
[E_sorted, sort_idx] = sort(E_emit(:));

%% Semiconductor parameters
NC = 8.63e13 * TK^(3/2);
NV = 1.83e15 * TK^(3/2);

%% Recombination parameters
C_auger = 15e-30;
B0      = 2.5e-10;

%% ===== DEFINE BASE (FIXED) PARAMETERS =====
% These are reasonable middle values from the training ranges
base_p0         = 1e18;           % middle of log-range [1e16, 1e19]
base_D          = 125;            % middle of [50, 200]
base_S          = 1e6;            % high SRV (near top of log-range [200, 1e7]) so the
                                   % surface recombination dip is deep enough to reveal
                                   % a peak on every curve, including the D/tau sweeps
base_tau        = 5e-8;           % middle of log-range [5e-9, 2.5e-7]
base_alpha_scale = 1;             % middle of log-range [0.1, 10]

%% ===== DEFINE VARIATION RANGES (4 values for most, 7 for p0) =====
n_vary = 4; % number of values per parameter (except p0)

% For logspace parameters, use logspace within the original range
p0_values       = logspace(log10(1e16),  log10(1e19),  7);    % 7 values for p0
D_values        = linspace(50,           200,          n_vary);
S_values        = logspace(log10(200),   log10(1e7),   n_vary);
tau_values      = logspace(log10(5e-9),  log10(2.5e-7), n_vary);
alpha_values    = logspace(log10(0.1),   log10(10),    n_vary);

%% ===== COMPUTE SELE FOR EACH PARAMETER VARIATION =====
% Storage: (n_vary or 7, x_res)
SELE_p0_var       = zeros(7, x_res);
SELE_D_var        = zeros(n_vary, x_res);
SELE_S_var        = zeros(n_vary, x_res);
SELE_tau_var      = zeros(n_vary, x_res);
SELE_alpha_var    = zeros(n_vary, x_res);
Ln_p0_var_um      = zeros(7, 1);       % diffusion length per p0 curve (microns)
Ln_D_var_um       = zeros(n_vary, 1);  % diffusion length per D curve (microns)
Ln_S_var_um       = zeros(n_vary, 1);  % diffusion length per S curve (microns)
Ln_tau_var_um     = zeros(n_vary, 1);  % diffusion length per tau curve (microns)
Ln_alpha_var_um   = zeros(n_vary, 1);  % diffusion length per alpha_scale curve (microns)
tau_eff_var_s     = zeros(n_vary, 1);  % effective lifetime per tau curve (s) -- differs from
                                        % input tau since tau_eff also mixes in tau_rad, tau_auger

fprintf('Computing sensitivity curves...\n');

for i = 1:7
    p0 = p0_values(i);
    [SELE_p0_var(i,:), ~, tau_eff_p0] = compute_sele_curve(p0, base_D, base_S, base_tau, base_alpha_scale, ...
        x, wavelength_PL, E_sorted, sort_idx, NC, NV, Kb, TK, C_auger, B0, ...
        n_k_wavelength, n_w_drude, k_w_drude, k_no_drude);
    Ln_p0_var_um(i) = sqrt(base_D * tau_eff_p0) * 1e4;
    fprintf('  p0 iteration %d/7 complete\n', i);
end

for i = 1:n_vary
    D = D_values(i);
    [SELE_D_var(i,:), ~, tau_eff_D] = compute_sele_curve(base_p0, D, base_S, base_tau, base_alpha_scale, ...
        x, wavelength_PL, E_sorted, sort_idx, NC, NV, Kb, TK, C_auger, B0, ...
        n_k_wavelength, n_w_drude, k_w_drude, k_no_drude);
    Ln_D_var_um(i) = sqrt(D * tau_eff_D) * 1e4;

    S = S_values(i);
    [SELE_S_var(i,:), ~, tau_eff_S] = compute_sele_curve(base_p0, base_D, S, base_tau, base_alpha_scale, ...
        x, wavelength_PL, E_sorted, sort_idx, NC, NV, Kb, TK, C_auger, B0, ...
        n_k_wavelength, n_w_drude, k_w_drude, k_no_drude);
    Ln_S_var_um(i) = sqrt(base_D * tau_eff_S) * 1e4;

    tau = tau_values(i);
    [SELE_tau_var(i,:), ~, tau_eff_tau] = compute_sele_curve(base_p0, base_D, base_S, tau, base_alpha_scale, ...
        x, wavelength_PL, E_sorted, sort_idx, NC, NV, Kb, TK, C_auger, B0, ...
        n_k_wavelength, n_w_drude, k_w_drude, k_no_drude);
    Ln_tau_var_um(i) = sqrt(base_D * tau_eff_tau) * 1e4;
    tau_eff_var_s(i) = tau_eff_tau;

    alpha_scale = alpha_values(i);
    [SELE_alpha_var(i,:), ~, tau_eff_alpha] = compute_sele_curve(base_p0, base_D, base_S, base_tau, alpha_scale, ...
        x, wavelength_PL, E_sorted, sort_idx, NC, NV, Kb, TK, C_auger, B0, ...
        n_k_wavelength, n_w_drude, k_w_drude, k_no_drude);
    Ln_alpha_var_um(i) = sqrt(base_D * tau_eff_alpha) * 1e4;

    fprintf('  D/S/tau/alpha_scale iteration %d/%d complete\n', i, n_vary);
end

%% ===== PLOTTING =====
fig = figure('Position', [100, 100, 1400, 900]);
set(fig, 'Color', 'white');

% Okabe-Ito colorblind-safe palette (distinguishable for deuteranopia,
% protanopia, and tritanopia). Order chosen to maximize contrast between
% adjacent curves.
okabe_ito = [ ...
    0,   114, 178;  % blue
    230, 159, 0;    % orange
    0,   158, 115;  % bluish green
    213, 94,  0;    % vermillion
    86,  180, 233;  % sky blue
    204, 121, 167;  % reddish purple
    240, 228, 66;   % yellow
    0,   0,   0     % black
    ] / 255;

% Distinct line styles/markers as a second visual cue beyond color
line_styles  = {'-', '--', '-.', ':', '-', '--', '-.'};
markers      = {'o', 's', '^', 'd', 'v', 'p', 'h'};

colors_4 = okabe_ito(1:4, :);
colors_7 = okabe_ito(1:7, :);

x_um = x * 1e4;

% Shared style for the peak-annotation labels (curve number / Ln) so they
% stay legible against overlapping curves, markers, and the legend box.
label_bg_opts = {'BackgroundColor', 'white', 'EdgeColor', [0.6 0.6 0.6], 'Margin', 1};

% Label text uses the curve color for quick visual association, but Okabe-Ito
% yellow is unreadably low-contrast on the white label background -- darken
% just the label text color for that curve (curve/legend keep true yellow).
label_colors_7 = colors_7;
label_colors_7(7,:) = [0.55, 0.5, 0];

% --- Subplot 1: p0 variation (7 curves) ---
subplot(2, 3, 1);
hold on;
for i = 1:7
    plot(x_um, SELE_p0_var(i,:), 'LineWidth', 2.0, 'Color', colors_7(i,:), ...
        'LineStyle', line_styles{i}, 'Marker', markers{i}, ...
        'MarkerIndices', 1:40:x_res, 'MarkerSize', 6, 'MarkerFaceColor', colors_7(i,:), ...
        'DisplayName', sprintf('%d: p0 = %.2e cm$^{-3}$', i, p0_values(i)));
    [peak_val, peak_idx] = max(SELE_p0_var(i,:));
    text(x_um(peak_idx), peak_val, sprintf('  %d ($L_n$=%.2f $\\mu$m)', i, Ln_p0_var_um(i)), ...
        'Color', label_colors_7(i,:), 'Interpreter', 'latex', 'FontWeight', 'bold', 'FontSize', 8, ...
        'VerticalAlignment', 'bottom', label_bg_opts{:});
end
hold off;
xlabel('Depth ($\mu$m)', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('SELE', 'Interpreter', 'latex', 'FontSize', 11);
title('Sensitivity to Doping Concentration (p0)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
grid on; grid minor;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);
legend('Interpreter', 'latex', 'FontSize', 9, 'Location', 'best');
xlim([0, max(x_um)]);
ylim([0, max(SELE_p0_var(:))*1.1]);

% --- Subplot 2: D variation ---
subplot(2, 3, 2);
hold on;
for i = 1:n_vary
    plot(x_um, SELE_D_var(i,:), 'LineWidth', 2.0, 'Color', colors_4(i,:), ...
        'LineStyle', line_styles{i}, 'Marker', markers{i}, ...
        'MarkerIndices', 1:40:x_res, 'MarkerSize', 6, 'MarkerFaceColor', colors_4(i,:), ...
        'DisplayName', sprintf('D = %.1f cm$^2$/s', D_values(i)));
    [peak_val, peak_idx] = max(SELE_D_var(i,:));
    text(x_um(peak_idx), peak_val, sprintf('  $L_n$=%.2f $\\mu$m', Ln_D_var_um(i)), ...
        'Color', colors_4(i,:), 'Interpreter', 'latex', 'FontWeight', 'bold', 'FontSize', 8, ...
        'VerticalAlignment', 'bottom', label_bg_opts{:});
end
hold off;
xlabel('Depth ($\mu$m)', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('SELE', 'Interpreter', 'latex', 'FontSize', 11);
title('Sensitivity to Diffusion Coefficient (D)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
grid on; grid minor;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);
legend('Interpreter', 'latex', 'FontSize', 9, 'Location', 'best');
xlim([0, max(x_um)]);
ylim([0, max(SELE_D_var(:))*1.1]);

% --- Subplot 3: S variation ---
subplot(2, 3, 3);
hold on;
for i = 1:n_vary
    plot(x_um, SELE_S_var(i,:), 'LineWidth', 2.0, 'Color', colors_4(i,:), ...
        'LineStyle', line_styles{i}, 'Marker', markers{i}, ...
        'MarkerIndices', 1:40:x_res, 'MarkerSize', 6, 'MarkerFaceColor', colors_4(i,:), ...
        'DisplayName', sprintf('S = %.2e cm/s', S_values(i)));
    [peak_val, peak_idx] = max(SELE_S_var(i,:));
    text(x_um(peak_idx), peak_val, sprintf('  $L_n$=%.2f $\\mu$m', Ln_S_var_um(i)), ...
        'Color', colors_4(i,:), 'Interpreter', 'latex', 'FontWeight', 'bold', 'FontSize', 8, ...
        'VerticalAlignment', 'bottom', label_bg_opts{:});
end
hold off;
xlabel('Depth ($\mu$m)', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('SELE', 'Interpreter', 'latex', 'FontSize', 11);
title('Sensitivity to Surface Recombination Velocity (S)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
grid on; grid minor;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);
legend('Interpreter', 'latex', 'FontSize', 9, 'Location', 'best');
xlim([0, max(x_um)]);
ylim([0, max(SELE_S_var(:))*1.1]);

% --- Subplot 4: tau variation ---
subplot(2, 3, 4);
hold on;
for i = 1:n_vary
    plot(x_um, SELE_tau_var(i,:), 'LineWidth', 2.0, 'Color', colors_4(i,:), ...
        'LineStyle', line_styles{i}, 'Marker', markers{i}, ...
        'MarkerIndices', 1:40:x_res, 'MarkerSize', 6, 'MarkerFaceColor', colors_4(i,:), ...
        'DisplayName', sprintf('$\\tau$ = %.2e s ($\\tau_{eff}$ = %.2e s)', tau_values(i), tau_eff_var_s(i)));
    [peak_val, peak_idx] = max(SELE_tau_var(i,:));
    text(x_um(peak_idx), peak_val, sprintf('  $L_n$=%.2f $\\mu$m', Ln_tau_var_um(i)), ...
        'Color', colors_4(i,:), 'Interpreter', 'latex', 'FontWeight', 'bold', 'FontSize', 8, ...
        'VerticalAlignment', 'bottom', label_bg_opts{:});
end
hold off;
xlabel('Depth ($\mu$m)', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('SELE', 'Interpreter', 'latex', 'FontSize', 11);
title('Sensitivity to Lifetime ($\tau$)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
grid on; grid minor;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);
legend('Interpreter', 'latex', 'FontSize', 9, 'Location', 'best');
xlim([0, max(x_um)]);
ylim([0, max(SELE_tau_var(:))*1.1]);

% --- Subplot 5: alpha_scale variation ---
subplot(2, 3, 5);
hold on;
for i = 1:n_vary
    plot(x_um, SELE_alpha_var(i,:), 'LineWidth', 2.0, 'Color', colors_4(i,:), ...
        'LineStyle', line_styles{i}, 'Marker', markers{i}, ...
        'MarkerIndices', 1:40:x_res, 'MarkerSize', 6, 'MarkerFaceColor', colors_4(i,:), ...
        'DisplayName', sprintf('$\\alpha$-scale = %.2f', alpha_values(i)));
    [peak_val, peak_idx] = max(SELE_alpha_var(i,:));
    text(x_um(peak_idx), peak_val, sprintf('  $L_n$=%.2f $\\mu$m', Ln_alpha_var_um(i)), ...
        'Color', colors_4(i,:), 'Interpreter', 'latex', 'FontWeight', 'bold', 'FontSize', 8, ...
        'VerticalAlignment', 'bottom', label_bg_opts{:});
end
hold off;
xlabel('Depth ($\mu$m)', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('SELE', 'Interpreter', 'latex', 'FontSize', 11);
title('Sensitivity to Absorption Scale ($\alpha_{scale}$)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
grid on; grid minor;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);
legend('Interpreter', 'latex', 'FontSize', 9, 'Location', 'best');
xlim([0, max(x_um)]);
ylim([0, max(SELE_alpha_var(:))*1.1]);

% --- Subplot 6: Summary information ---
subplot(2, 3, 6);
axis off;
summary_text = sprintf( ...
    ['Base Parameter Values:\n', ...
     '\n', ...
     'p0 = %.2e cm^{-3}\n', ...
     'D = %.1f cm^2/s\n', ...
     'S = %.2e cm/s\n', ...
     'tau_eff = %.2e s\n', ...
     'alpha_scale = %.2f\n', ...
     '\n', ...
     'S is fixed high so every curve\n', ...
     'shows a subsurface peak. Ln\n', ...
     'labels give the diffusion length\n', ...
     'sqrt(D*tau_eff) for each curve.\n', ...
     '\n', ...
     'Parameter Ranges:\n', ...
     '\n', ...
     'p0: [1e16, 1e19] (logspace)\n', ...
     'D: [50, 200] (linear)\n', ...
     'S: [200, 1e7] (logspace)\n', ...
     'tau: [5e-9, 2.5e-7] (logspace)\n', ...
     'alpha_scale: [0.1, 10] (logspace)'], ...
    base_p0, base_D, base_S, base_tau, base_alpha_scale);
text(0.05, 0.95, summary_text, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
    'Interpreter', 'none', 'FontSize', 9, ...
    'BackgroundColor', 0.95*[1 1 1], 'EdgeColor', [0.7 0.7 0.7], ...
    'Margin', 10);

sgtitle('SELE Parameter Sensitivity Analysis', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');

pause(0.1);
