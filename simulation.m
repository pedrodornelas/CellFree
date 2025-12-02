clc
close all
clear all

% Projeto 3 - Comunicacoes Moveis - PPGEE 2025/2
% Autoria: Pedro Henrique Dornelas Almeida
% Cell-Free Massive MIMO Simulation

%% Global Parameters
fc = 3 * 1e9 ;  % 3 GHz
bw = 20 * 1e6 ; % 20 MHz
noise_figure_db = 9 ; % dB
noise_figure = 10^(noise_figure_db/10) ;
h_aps = 15 ;   % m
h_ues = 1.65 ; % m
T = 296.15 ;   % K
Lx = 1000 ;   % m
Ly = 1000 ;   % m

% Coherence block numbers
Nbc = 100 ;
% Total enviroments testeds
Ncf = 300 ;
% Pilot symbols power
Pp = 0.2 ; % W = 200 mW
% Downlink power
Pd = 0.2 ; % W = 200 mW
% Pilot length
tau_p = 50 ; % symbols

% noise power
k_boltzmann = 1.381e-23 ; % J/K
sigma2 = k_boltzmann * T * bw * noise_figure ; % W

%% APs and UEs distribution

% Number of APs and UEs
M = 100 ; % Number of APs
K = 20 ;  % Number of UEs

% loop of Ncf enviroments
for n = 1:Ncf

    % Generate random positions for APs
    % pAP,m = [xAP,m, yAP,m, hAP]^T
    pAP = zeros(3, M) ;
    pAP(1, :) = (rand(1, M) - 0.5) * Lx ; % xAP,m ~ U[-Lx/2, Lx/2]
    pAP(2, :) = (rand(1, M) - 0.5) * Ly ; % yAP,m ~ U[-Ly/2, Ly/2]
    pAP(3, :) = h_aps ;                    % hAP (constant height)

    % Generate random positions for UEs
    % pUE,k = [xUE,k, yUE,k, hUE]^T
    pUE = zeros(3, K) ;
    pUE(1, :) = (rand(1, K) - 0.5) * Lx ; % xUE,k ~ U[-Lx/2, Lx/2]
    pUE(2, :) = (rand(1, K) - 0.5) * Ly ; % yUE,k ~ U[-Ly/2, Ly/2]
    pUE(3, :) = h_ues ;                    % hUE (constant height)

    % Visualize the distribution of APs and UEs
    % figure('Name', 'AP and UE Distribution')
    % scatter(pAP(1, :), pAP(2, :), 100, 'b^', 'filled', 'DisplayName', 'APs')
    % hold on
    % scatter(pUE(1, :), pUE(2, :), 80, 'ro', 'filled', 'DisplayName', 'UEs')
    % grid on
    % xlabel('x [m]')
    % ylabel('y [m]')
    % title(sprintf('Cell-Free System: %d APs and %d UEs', M, K))
    % legend('Location', 'best')
    % axis equal
    % xlim([-Lx/2 Lx/2])
    % ylim([-Ly/2 Ly/2])
    % hold off

    % distance matrix d(m,k)
    d = zeros(M, K) ;
    for m = 1:M
        for k = 1:K
            % d(m,k) = || pAP,m - pUE,k ||
            d(m, k) = norm(pAP(:, m) - pUE(:, k));
        end
    end

    %% Fading Coefficients Calculation
    % Large Coefficients
    omega = zeros(M, K) ; % Large-scale fading coefficients matrix
    for m = 1:M
        for k = 1:K
            omega(m, k) = large_scale_fading(d(m, k), fc) ;
        end
    end

    % loop of Nbc coherence blocks
    for bc = 1:Nbc

        % Small Scale Fading Coefficients
        % h(m,k) ~ CN(0,1)
        h = (randn(M, K) + 1i * randn(M, K)) / sqrt(2) ;
        % channel
        g = h .* sqrt(omega);

        %% Channel Estimation Phase
        % equivalent noise
        noise = (randn(M, K) + 1i * randn(M, K)) / sqrt(2) * sqrt(sigma2) ;
        % received pilot signal at APs
        y = sqrt(tau_p * Pp) * g + noise ;
        % MMSE Channel Estimation
        cmk = (sqrt(tau_p * Pp) * omega) ./ (tau_p * Pp * omega + sigma2) ;
        g_hat = cmk .* y ;

        %% Downlink Data Transmission Phase
        % Total and Uniform Power are considered
        gamma_mk = ones(M, K) ;
        gamma_mk = sqrt(Pd .* tau_p) .* omega .* cmk ;
        % Power control coefficients
        eta_m = ones(M, 1) ;
        for m = 1:M
            eta_m(m) = 1 / sum(gamma_mk(m, :)) ;
        end

        % Expand to full matrix eta_mk (eta_mk(m,k) = eta_m(m))
        eta_mk = repmat(eta_m, 1, K) ;

        % SINR Calculation
        sinr_k_stat = sinr_stat(K, Pd, eta_mk, gamma_mk, omega, sigma2) ; % statistical SINR
        sinr_k_inst = sinr_inst(K, Pd, eta_mk, g, g_hat, sigma2) ; % instantaneous SINR - with known g_hat

    end % end of Nbc coherence blocks loop
end % end of Nfc enviroments loop

'end'


% Achievable spectral efficiencies (bits/s/Hz)
% rate_k = log2(1 + sinr_k_stat) ;
% avg_rate = mean(rate_k) ;
% fprintf('Computed SINR for %d users. Avg rate = %.4f bits/s/Hz\n', K, avg_rate) ;


%% Auxiliar functions
function omega = large_scale_fading(d_mk, fc)
    c = 3e8 ;           % speed of light (m/s)
    d0 = 1 ;            % reference distance (1 m)

    % avoid unphysical distances below 1 m by clamping
    d_eff = max(d_mk, d0) ;

    % Free-space path loss at 1 meter (dB)
    PLFS_1m = 20*log10(4*pi*d0*fc / c) ;

    % shadowing: zero-mean Gaussian in dB with sigma = 8 dB
    sigma_sf = 8 ;
    chi_sf = sigma_sf * randn() ;

    % CI model (path-loss exponent implicit in the 28 factor)
    Omega_dB = PLFS_1m + 28*log10(d_eff) + chi_sf ;

    % convert to linear scale
    omega = 10^(Omega_dB/10) ;
end

function sinr_k = sinr_stat(K, Pd, eta_mk, gamma_mk, omega, sigma2)
    % Statistical SINR calculation
    sinr_k = zeros(K, 1) ;
    for k = 1:K
        % coherent desired sum across APs for user k
        s_sum = sum( sqrt(eta_mk(:, k)) .* gamma_mk(:, k) ) ;
        numerator = Pd * ((s_sum) .^ 2) ;

        % interference: sum over all users k' and APs m, weighted by omega(m,k)
        interference = 0 ;
        for kp = 1:K
            interference = interference + sum( eta_mk(:, kp) .* gamma_mk(:, kp) .* omega(:, k) ) ;
        end

        denominator = (Pd .* interference) + sigma2 ;

        sinr_k(k) = numerator / denominator ;
    end
end

function sinr_k = sinr_inst(K, Pd, eta_mk, g, g_hat, sigma2)
    % Instantaneous SINR calculation
    sinr_k = zeros(K, 1) ;
    for k = 1:K
        % coherent desired sum across APs for user k
        s_sum = sum( sqrt(eta_mk(:, k)) .* g(:, k) .* conj(g_hat(:, k)) ) ;
        numerator = Pd * ((abs(s_sum)) .^ 2) ;

        % interference: sum over all users k' and APs m, weighted by omega(m,k)
        interference = 0 ;
        for kp = 1:K
            if kp == k
                continue ; % skip desired user
            end
            interference = interference + sum( abs(sqrt(eta_mk(:, kp)) .* g(:, kp) .* conj(g_hat(:, k))) .^ 2 ) ;
        end

        denominator = (Pd .* interference) + sigma2 ;

        sinr_k(k) = numerator / denominator ;
    end
end