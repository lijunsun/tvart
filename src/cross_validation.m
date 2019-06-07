clear all
close all;
clc;

X = load('traffic.mat','data');
X = X.data;
days = 30;
X = X(:,1:144*days);
M = 1;
R = 10;
max_iter = 100;   

addpath('C:\Users\sunli\Dropbox\CommonMatlab\unlocbox')
init_unlocbox();
warning('on');


test_X = load('traffic.mat','data');
test_X = test_X.data;
test_X = test_X(:,144*days+1:144*days+1+2000);

%%
clc;
load('4000_beta_1000_eta_0.01.mat');
init.A = A;
init.B = B;
init.C = C(1:size(test_X,2)-1,:);
%init.C = C;
warning('on');

%%
beta = 1000;
eta1 = 0.01;
R = 10;
max_iter = 200;
center = 1;
[lambda, A, B, C, cost, Xten, Yten, rmse] = TVART_alt_min_Lijun(test_X, M, R, ...
                  'center', center, ...
                  'eta', eta1, ...
                  'beta', beta, ...
                  'regularization', 'spline', ...
                  'verbosity', 2, ...
                  'max_iter', max_iter, ...
                  'init', init);
                  %'init', false);