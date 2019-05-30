clear all
close all
clc;
set(0,'DefaultFigurePaperPositionMode','auto')

%set(0,'DefaultFigurePosition',[25,25,500,500]);
set(0,'DefaultAxesFontSize',10)
set(0,'DefaultAxesFontName','Helvetica')
set(0,'DefaultAxesLineWidth',1);

set(0,'DefaultTextFontSize',12)
set(0,'DefaultTextFontName','Helvetica')
set(0,'DefaultLineLineWidth',1);

addpath('~/work/MATLAB/unlocbox')
init_unlocbox();

rng(1);

data_dir = '../data_full/20090525S1_FTT_K1_ZenasChao_mat_ECoG64-Motion8';

figdir = '../figures/';
post_ecog_data = [data_dir '/Kam_Bands.mat'];
%post_ecog_data= [data_dir '/Kam_Bands_post_Steve.mat']);
t0 = 200;
tf = 1000;
center = 0;
voltage = 1;
regularization = 'TV';
save_file = '../data/test_neurotycho.mat';

load([data_dir '/Motion.mat']);
ntime = size(MotionData{1}, 1);
ncoord = size(MotionData{1}, 2);
nchan = length(MotionData);
X = zeros(ntime, ncoord*nchan);
for i = 1:nchan
    X(:, (i-1) * ncoord + 1 : i * ncoord) = MotionData{i};
end
X = X';
X = standardize(X);

idx = (MotionTime >= t0 & MotionTime <= tf);
MotionTime = MotionTime(idx);
X = X(:, idx);

M = 30;
R = 8;
%% tensor DMD algorithm TV
max_iter = 1000;             % iterations
eta1 = 0.005;               % Tikhonov
eta2 = 0.01;               % regularization
beta = 0.15;               % TV-regularization
%% TV-l0
max_iter = 1000;             % iterations
eta1 = 0.01;               % Tikhonov
eta2 = 0.01;               % regularization
beta = 0.01;               % TV-regularization
%% ALS
nu = 0.0;    % ALS noise level
RALS = 1;


% [lambda, A, B, C, Xten, Yten] = ...
%     tensor_DMD_ALS(X', M, R, ...
%                    'center', 1, ...
%                    'eta', eta, ...
%                    'nu', nu, ...
%                    'beta', beta, ...
%                    'max_iter', max_iter);

% [lambda, A, B, C, cost, Xten, Yten, rmse_vec, W] = ...
%     tensor_DMD_ALS_aux(X, M, R, ...
%                        'center', center, ...
%                        'eta1', eta1, ...
%                        'eta2', eta2, ...
%                        'beta', beta, ...
%                        'regularization', 'TV', ...
%                        'rtol', 1e-5, ...
%                        'max_iter', max_iter);

[lambda, A, B, C, cost, Xten, Yten, rmse_vec] = ...
    TVART_alt_min(X, M, R, ...
                  'center', center, ...
                  'eta', eta1, ...
                  'beta', beta, ...
                  'regularization', regularization, ...
                  'verbosity', 2, ...
                  'max_iter', max_iter);
%W = C;

[lambda_r, A_r, B_r, C_r] = rebalance(A, B, C, 1);
[lambda_r, A_r, B_r, C_r] = reorder_components(lambda_r, A_r, B_r, ...
                                                  C_r);
%A_r = A; B_r = B; C_r = C; lambda_r = lambda;
%C_r = C_r * diag(lambda_r);

%% Old ECoG preprocessing, now uses preprocess_neurotycho.py
% ecog_time = load([data_dir '/ECoG_time.mat']);
% ecog_time = ecog_time.('ECoGTime');
% n_chan = 64;
% ecog_data = zeros(n_chan, length(ecog_time));
% for chan = 1:n_chan
%     fn = sprintf('%s/ECoG_ch%d.mat', data_dir, chan);
%     var = sprintf('ECoGData_ch%d', chan);
%     tmp = load(fn);
%     ecog_data(chan, :) = tmp.(var);
% end
% %% Common average referencing
% ecog_data = ecog_data - repmat(mean(ecog_data, 1), n_chan, 1);
% %ecog_data = ecog_data - repmat(mean(ecog_data, 2), 1, length(ecog_data));
% %ecog_data = ecog_data./ repmat(std(ecog_data, 0, 2), 1,
% %length(ecog_data));
% ecog_data = standardize(ecog_data);
% DT = ecog_time(2) - ecog_time(1);

fprintf('\nNow working on ECoG data\n')

load(post_ecog_data)
%voltage = 0;
% %% Miller comparison
%load([data_dir '/Kam_Bands_param_Miller.mat'])
%ecog_data = [ecog_post(1:3:end, :) ;
%             ecog_post(3:3:end, :) ];
% M_ecog = 6;
% R_ecog = 3;
% eta2 = 1e-4;
% beta_ecog = 3e2;
if voltage
    ecog_data = ecog_v_filtered;
    ecog_time = ecog_v_time;
    % Voltage params - TV!
    M_ecog = 200;
    R_ecog = 8;
    eta1 = 1.0;
    eta2 = 1e-4;
    beta_ecog = 100;
    % Voltage params - TV-L0
    % M_ecog = 200;
    % R_ecog = 10;
    % eta1 = 1.;
    % eta2 = 5e-5;
    % beta_ecog = 2.0;
    % downsample
    ecog_data = ecog_data(:, 1:2:end);
    ecog_time = ecog_time(1:2:end);
else
    %% Power params - TV!
    M_ecog = 12;
    R_ecog = 10;
    eta1 = 1.;
    eta2 = 1e-4;
    beta_ecog = 300;
    ecog_data = ecog_power_post;
    ecog_time = ecog_power_time;
end

ecog_data = ecog_data(:, ecog_time <= MotionTime(end) & ecog_time >= MotionTime(1));
ecog_data = standardize(ecog_data);
ecog_time = ecog_time(ecog_time <= MotionTime(end) & ecog_time >= MotionTime(1));



% [ecog_lambda, ecog_A, ecog_B, ecog_C, ecog_cost, ~,~, ecog_rmse, ecog_W] = ...
%     tensor_DMD_ALS_aux(ecog_data, M_ecog, R_ecog, ...
%                        'center', center, ...
%                        'eta1', eta1, ...
%                        'eta2', eta2, ...
%                        'beta', beta_ecog, ...
%                        'regularization', 'TV', ...
%                        'rtol', 1e-4,...
%                        'max_iter', max_iter);

% [ecog_lambda, ecog_A, ecog_B, ecog_C, ecog_cost, ~,~, ecog_rmse] = ...
%     tensor_DMD_alt_min(ecog_data, M_ecog, R_ecog, ...
%                        'center', center, ...
%                        'eta', eta1, ...
%                        'beta', beta_ecog, ...
%                        'regularization', 'TV', ...
%                        'rtol', 1e-5,...
%                        'verbosity', 2,...
%                        'max_iter', max_iter);


[ecog_lambda, ecog_A, ecog_B, ecog_C, ecog_cost, ~,~, ecog_rmse] = ...
    TVART_alt_min(ecog_data, M_ecog, R_ecog, ...
                  'center', center, ...
                  'eta', eta1, ...
                  'beta', beta_ecog, ...
                  'regularization', regularization, ...
                  'verbosity', 2,...
                  'max_iter', max_iter); 


                                         %'init', struct('A',
                                         %ecog_A, 'B', ecog_B, 'C',
                                         %ecog_C));
                                         
[ecog_lambda, ecog_A, ecog_B, ecog_C, ecog_cost, ~,~, ecog_rmse] = ...
    TVART_alt_min(ecog_data, M_ecog, R_ecog, ...
                  'center', center, ...
                  'eta', eta1, ...
                  'beta', beta_ecog, ...
                  'regularization', regularization, ...
                  'verbosity', 2,...
                  'max_iter', max_iter, ...
                  'rtol', 1e-4,...
                  'init', struct('A', ecog_A, 'B', ecog_B, 'C', ...
                                 ecog_C));

ecog_W = ecog_C;
ecog_A_r = ecog_A; ecog_B_r = ecog_B; ecog_C_r = ecog_C; ...
     ecog_lambda_r = ecog_lambda; 
% [ecog_lambda_r, ecog_A_r, ecog_B_r, ecog_C_r, ecog_W_r] = ...
%     rebalance_2(ecog_A, ecog_B, ecog_C, ecog_W, 1);
% [ecog_lambda_r, ecog_A_r, ecog_B_r, ecog_C_r] = ...
%      rebalance(ecog_A, ecog_B, ecog_C, 1);
% [ecog_lambda_r, ecog_A_r, ecog_B_r, ecog_C_r] = ...
%     reorder_components(ecog_lambda_r, ecog_A_r, ecog_B_r, ecog_C_r);
% ecog_C_r = ecog_C_r * diag(ecog_lambda_r);

% s = struct('A', ecog_A, 'B', ecog_B, 'C', ecog_C, 'lambda', ecog_lambda, ...
%            'W', ecog_W);

% [ecog_lambda, ecog_A, ecog_B, ecog_C, ecog_cost, ~,~, ecog_rmse, ecog_W] = ...
%     tensor_DMD_ALS_aux(ecog_data, M_ecog, R_ecog, ...
%                        'center', 0, ...
%                        'eta1', eta1, ...
%                        'eta2', eta2, ...
%                        'beta', beta_ecog, ...
%                        'regularization', 'TV', ...
%                        'max_iter', max_iter, ...
%                        'init', s);

% [ecog_lambda, ecog_A, ecog_B, ecog_C, ecog_cost, ~,~, ecog_rmse] = ...
%      tensor_DMD_prox_grad(ecog_data, M_ecog, R_ecog, ...
%                           'center', 0, ...
%                           'eta', eta1, ...
%                           'step_size', 1e-6, ...
%                           'beta', beta_ecog, ...
%                           'iter_disp', 1, ...
%                           'max_iter', 1000, ...
%                           'init', s );         


% eta2 = 1e-3;
% [ecog_lambda, ecog_A, ecog_B, ecog_C, ecog_cost, ~,~, ecog_rmse] = ...
%     tensor_DMD_ALS_smooth(ecog_post, M_ecog, R_ecog, ...
%                           'center', 0, ...
%                           'eta1', eta1, ...
%                           'eta2', eta2, ...
%                           'beta', 5, ...
%                           'max_iter', 300);


% figure;
% ax1 = subplot(2,1,1);
% plot(MotionTime(1:M:end-M), sum(C.^2, 2))
% title('MOCAP extracted components')
% axis tight
% ax2 = subplot(2,1,2);
% plot(ecog_time(1:M_ecog:end-M_ecog), sum(ecog_C.^2, 2))
% title('ECoG extracted components')
% xlabel('time (s)')
% axis tight
% linkaxes([ax1 ax2], 'x')
% xlim([100, 300])

ts1 = timeseries(X, MotionTime);
ts3 = timeseries(ecog_C, ecog_time(1:M_ecog:end-M_ecog));
ts2 = timeseries(C, MotionTime(1:M:end-M));
[ts2,ts3] = synchronize(ts2, ts3, 'union');
% tmp=corrcoef([ts2.Data, ts3.Data]); 
% disp(tmp);

% figure;
% semilogy(ecog_cost)
% title('ECoG convergence')

% figure;
% semilogy(cost)
% title('MOCAP convergence')

clusters_mocap = kmeans(X', 2);
clusters_mocap_modes = kmedoids(C_r, 2);
clusters_ecog_modes = kmedoids(ecog_C_r, 2);

% figure;
% ax1 = subplot(3,1,1);
% plot(MotionTime(1:M:end-M), C_r)
% title('MOCAP extracted components')
% axis tight;
% ax2 = subplot(3,1,2);
% plot(MotionTime(1:M:end-M), clusters, 'o')
% ylim([0,4]);
% title('MOCAP clusters')
% ax3 = subplot(3,1,3);
% plot(ecog_time(1:M_ecog:end-M_ecog), ecog_clusters, 'o')
% ylim([0,4]);
% title('ECoG clusters')
% %plot(transitions(:,1)/M, transitions(:,2)+1, 'k')
% linkaxes([ax1 ax2 ax3], 'x')
% xlim([120, 320])


%% correlate ecog signals against movement`

lf_power = sum(ecog_power_post(1:2:end,:),1);
hf_power = sum(ecog_power_post(2:2:end,:),1);

% [~, locs] = findpeaks(X(6,:), 'MinPeakWidth', 20, ...
%                      ' MinPeakProminence', 3);

idx_changes = findchangepts(X, 'MaxNumChanges', 120);


%% Save output
save(save_file);


%% Now plotting


figure;
ax1 = subplot(5,1,1:2);
plot(MotionTime, X,  'color', [0,0,0] + 0.4, 'linewidth', 0.5);
vline(MotionTime(idx_changes), 'k--')
ylim([-8 8])
title('MOCAP data', 'fontweight','bold')
%grid on; grid minor
set(gca, 'xticklabels', {})
% ax2 = subplot(8,1,7);
% %plot(MotionTime, clusters_mocap, 'o');
% imagesc([MotionTime(1), MotionTime(end)], [0,1], clusters_mocap');
% colormap(gca, 'winter')
% ylabel('MOCAP', 'fontweight','bold')
% %ylim([0.9, 2.1])
% %grid on; grid minor
% %set(gca, 'xtick', [])
% set(gca, 'ytick', [])
% set(gca, 'xticklabels', {})
ax2 = subplot(5,1,3);
%plot(MotionTime(1:M:end-M), clusters_mocap_modes, 'o')
%plot(ecog_time(1:M_ecog:end-M_ecog), clusters_ecog_modes,'o');
imagesc([ecog_time(1), ecog_time(end-M_ecog)], [0 1], clusters_ecog_modes');
colormap(gca, 'summer')
%set(gca, 'xtick', [])
set(gca, 'ytick', [])
set(gca, 'xticklabels', {})
title('ECoG clusters', 'fontweight','bold')
ax3 = subplot(5,1,4);
plot(ecog_power_time, lf_power)
hold on; 
plot(ecog_power_time, hf_power,'r-')
grid on; grid minor
legend({'low freq', 'high freq'})
title('ECoG band power', 'fontweight', 'bold')
axis tight
set(gca, 'xticklabels', {})
ax5 = subplot(5,1,5);
%tmp = ecog_C_r - repmat(mean(ecog_C_r,1), size(ecog_C_r,1), 1);
tmp = ecog_C_r;
%tmp = ecog_C_r(:,2) - ecog_C_r(:,1);
plot(ecog_time(1:M_ecog:(length(tmp)*M_ecog)), tmp); 
%vline(MotionTime(clusters_mocap == 1), 'k:');
%vline(MotionTime(locs), 'k:');
title('Temporal modes', 'fontweight', 'bold')
axis tight
%ylim([0.9, 2.1])
%grid on; grid minor
xlabel('Time (s)')
linkaxes([ax1 ax2 ax3 ax5], 'x')
xlim([700 800])
set(gcf,'renderer','Painters')
set(gcf, 'Color', 'w', 'Position', [100 200 600 700]);
set(gcf, 'PaperUnits', 'inches', ...
         'PaperPosition', [0 0 6.5 7.5], 'PaperPositionMode', 'manual');
print('-depsc2', '-loose', [figdir 'neurotycho_clusters.eps'], '-r300');


if ~voltage
    figure;
    ax1 = subplot(4,1,1);
    plot(MotionTime, X)
    grid on; grid minor
    %vline(MotionTime(clusters_mocap == 1), 'k:');
    title('MOCAP data', 'fontweight', 'normal')
    axis tight;
    ax2 = subplot(4,1,2);
    plot(MotionTime(1:M:end-M), C_r)
    grid on; grid minor
    title('MOCAP temporal modes', 'fontweight', 'normal')
    axis tight
    ax4 = subplot(4,1,3);
    plot(ecog_power_time, lf_power)
    hold on; 
    plot(ecog_power_time, hf_power,'r-')
    grid on; grid minor
    legend({'low freq', 'high freq'})
    title('ECoG band power', 'fontweight', 'normal')
    axis tight
    ax3 = subplot(4,1,4);
    %tmp = ecog_C_r - repmat(mean(ecog_C_r,1), size(ecog_C_r,1), 1);
    tmp = ecog_C_r;
    %tmp = ecog_C_r(:,2) - ecog_C_r(:,1);
    plot(ecog_time(1:M_ecog:(length(tmp)*M_ecog)), tmp); 
    grid on; grid minor
    %vline(MotionTime(clusters_mocap == 1), 'k:');
    %vline(MotionTime(locs), 'k:');
    title('ECoG power: temporal modes', 'fontweight', 'normal')
    xlabel('time (s)')
    legend
    axis tight
    linkaxes([ax1 ax2 ax3 ax4], 'x')
    xlim([t0, tf])
    %xlim([200, 300])
    set(gcf,'renderer','Painters')
    set(gcf, 'Color', 'w', 'Position', [100 500 800 980]);
    set(gcf, 'PaperUnits', 'inches', ...
             'PaperPosition', [0 0 7.5 10], ...
             'PaperPositionMode', 'manual');
    print('-depsc2', '-loose', [figdir 'neurotycho_summary.eps'], '-r300');
else
    figure;
    ax1 = subplot(4,1,1);
    plot(MotionTime, X)
    grid on; grid minor
    %vline(MotionTime(clusters_mocap == 1), 'k:');
    title('MOCAP data')
    axis tight;
    ax2 = subplot(4,1,2);
    plot(MotionTime(1:M:end-M), C_r)
    grid on; grid minor
    title('MOCAP temporal modes')
    axis tight
    ax4 = subplot(4,1,3);
    plot(ecog_power_time, lf_power)
    hold on; 
    plot(ecog_power_time, hf_power,'r-')
    grid on; grid minor
    legend({'low freq', 'high freq'})
    title('ECoG low / high power')
    axis tight
    ax3 = subplot(4,1,4);
    %tmp = ecog_C_r - repmat(mean(ecog_C_r,1), size(ecog_C_r,1), 1);
    tmp = ecog_C_r;
    %tmp = ecog_C_r(:,2) - ecog_C_r(:,1);
    plot(ecog_time(1:M_ecog:(length(tmp)*M_ecog)), tmp); 
    grid on; grid minor
    %vline(MotionTime(clusters_mocap == 1), 'k:');
    %vline(MotionTime(locs), 'k:');
    title('ECoG voltage: temporal modes')
    xlabel('time (s)')
    %legend
    axis tight
    linkaxes([ax1 ax2 ax3 ax4], 'x')
    %xlim([t0, tf])
    xlim([700, 760])
    set(gcf,'renderer','Painters')
    set(gcf, 'Color', 'w', 'Position', [0 0 750 1000]);
    set(gcf, 'PaperUnits', 'inches', ...
             'PaperPosition', [0 0 7.5 10], ...
             'PaperPositionMode', 'manual');
    print('-depsc2', '-loose', [figdir 'neurotycho_summary.eps'], '-r200');
end




%% Plot modes
if ~voltage
    %% Plot modes - LOW/HIGH power
    addpath('~/work/MATLAB')
    img = imread([figdir 'monkeys/monkey_K1.png']);
    load('../data/K1_electrodes.mat')
    for mode = 1:R_ecog
        figure;
        ax = subplot(1,4,1);
        image(img);
        set(gca,'YTickLabel',[]);
        set(gca, 'XtickLabel', []);
        hold on
        vec = ecog_A_r(:, mode);
        scatter(img_electrodes_x, img_electrodes_y, 80, vec(1:2:end), ...
                'filled');
        colormap(flipud(brewermap([], 'PuOr')))
        colorbar
        symmetrize_colorbar();
        title('Low freq, left')
        ylabel(sprintf('Mode %d', mode), 'fontsize', 20)
        axis equal
        axis tight

        ax= subplot(1,4,2);
        image(img);
        set(gca,'YTickLabel',[]);
        set(gca, 'XtickLabel', []);
        hold on
        vec = ecog_A_r(:, mode);
        scatter(img_electrodes_x, img_electrodes_y, 80, vec(2:2:end), 'filled');
        colormap(flipud(brewermap([], 'PuOr')))
        colorbar
        symmetrize_colorbar();
        title('High freq, left')
        axis equal
        axis tight

        ax = subplot(1,4,3);
        image(img);
        set(gca,'YTickLabel',[]);
        set(gca, 'XtickLabel', []);
        hold on
        if center
            vec = ecog_B_r(1:end-1, mode);
        else
            vec = ecog_B_r(:, mode);
        end
        scatter(img_electrodes_x, img_electrodes_y, 80, vec(1:2:end), 'filled');
        colormap(flipud(brewermap([], 'PuOr')))
        colorbar
        symmetrize_colorbar();
        title('Low freq, right')
        axis equal
        axis tight

        ax = subplot(1,4,4);
        image(img);
        set(gca,'YTickLabel',[]);
        set(gca, 'XtickLabel', []);
        hold on
        if center
            vec = ecog_B_r(1:end-1, mode);
        else
            vec = ecog_B_r(:, mode);
        end
        scatter(img_electrodes_x, img_electrodes_y, 80, vec(2:2:end), 'filled');
        colormap(flipud(brewermap([], 'PuOr')))
        colorbar
        symmetrize_colorbar();
        title('High freq, right')
        axis equal
        axis tight
        if center
            xlabel(sprintf('affine weight = %1.2g, %1.2g%%', ...
                           ecog_B_r(end, mode), ...
                           abs(ecog_B_r(end, mode)) / sum(abs(ecog_B_r(end,:))) * 100 ),...
                   'fontsize', 14);
        end

        set(gcf,'renderer','Painters')
        set(gcf, 'Color', 'w')
        set(gcf, 'position', [ 105         450        1653 ...
                            473])
        set(gcf, 'PaperUnits', 'inches', ...
                 'PaperPosition', [0 0 11 3.2], ...
                 'PaperPositionMode', 'auto');
        print('-depsc2', '-loose', '-r300', ...
              sprintf('%sneurotycho_mode_%d.eps', figdir, mode));
    end
else
    %% Plot modes - VOLTAGE
    addpath('~/work/MATLAB')
    img = imread([figdir 'monkeys/monkey_K1.png']);
    load('../data/K1_electrodes.mat')
    for mode = 1:R_ecog
        figure;
        ax = subplot(1,2,1);
        image(img);
        set(gca,'YTickLabel',[]);
        set(gca, 'XtickLabel', []);
        hold on
        vec = ecog_A_r(:, mode);
        scatter(img_electrodes_x, img_electrodes_y, 80, vec, ...
                'filled');
        colormap(flipud(brewermap([], 'PuOr')))
        colorbar
        symmetrize_colorbar();
        title('Left')
        ylabel(sprintf('Mode %d', mode), 'fontsize', 20)
        axis equal
        axis tight

        ax = subplot(1,2,2);
        image(img);
        set(gca,'YTickLabel',[]);
        set(gca, 'XtickLabel', []);
        hold on
        if center
            vec = ecog_B_r(1:end-1, mode);
        else
            vec = ecog_B_r(:, mode);
        end
        scatter(img_electrodes_x, img_electrodes_y, 80, vec, 'filled');
        colormap(flipud(brewermap([], 'PuOr')))
        colorbar
        symmetrize_colorbar();
        title('Right')
        axis equal
        axis tight
        if center
            xlabel(sprintf('affine weight = %1.2g, %1.2g%%', ...
                           ecog_B_r(end, mode), ...
                           abs(ecog_B_r(end, mode)) / sum(abs(ecog_B_r(end,:))) * 100 ), ...
                   'fontsize', 14);
        end

        set(gcf,'renderer','Painters')
        set(gcf, 'Color', 'w')
        set(gcf, 'position', [ 105         450        830 ...
                            473])
        set(gcf, 'PaperUnits', 'inches', ...
                 'PaperPosition', [0 0 5.5 3.2], ...
                 'PaperPositionMode', 'auto');
        print('-depsc2', '-loose', '-r300', ...
              sprintf('%sneurotycho_mode_%d.eps', figdir, mode));
    end
end
