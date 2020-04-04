clear all;close all;clc;
%% Obrada .mat fajlova
% PEOCESSING DIRECTORY 01
directory = 'C:\Users\user\Desktop\SPC\01';
[~,~,~,~,~,~,global_position_rel_alt1,global_position_compass_hdg1,...
    local_position_pose1,local_position_odom1] = process_mat_files(directory);
% PEOCESSING DIRECTORY 02
directory = 'C:\Users\user\Desktop\SPC\02';
[~,~,~,~,~,~,global_position_rel_alt2,global_position_compass_hdg2,...
    local_position_pose2,local_position_odom2] = process_mat_files(directory);
% PROCESSING DIRECTORY 03
directory = 'C:\Users\user\Desktop\SPC\03'; % full path of folder with mat files
[frames3,IMU_data3,IMU_diff_pressure3,IMU_mag3,IMU_static_pressure3,...
    IMU_temperature_baro3,~,global_position_compass_hdg3,~,~] = process_mat_files(directory);
% PROCESSING DIRECTORY 04
directory = 'C:\Users\user\Desktop\SPC\04'; % full path of folder with mat files
[frames4,IMU_data4,IMU_diff_pressure4,IMU_mag4,IMU_static_pressure4,...
    IMU_temperature_baro4,~,global_position_compass_hdg4,~,~] = process_mat_files(directory);
%==========================================================================
%% 1. VIZUELIZACIJA 
%obrada frejmova
movie3 = raw_frames_to_movie(frames3);
movie4 = raw_frames_to_movie(frames4);
%% pustanje snimka
%implay(squeeze(movie3(3,:,:,:)));
implay(squeeze(movie4(2,:,:,:)));
%==========================================================================
%% 2. STRUKTURA PODATAKA
% global position
global_position_compass_hdg1 = cell2mat(global_position_compass_hdg1.bMsg);
global_position_compass_hdg1 = [global_position_compass_hdg1(:).Data];
global_position_compass_hdg2 = cell2mat(global_position_compass_hdg2.bMsg);
global_position_compass_hdg2 = [global_position_compass_hdg2(:).Data];
% prikaz podataka
figure(1);
subplot(2,1,1);
plot(global_position_compass_hdg1);
title('compass heading in degrees(regular)');
hold on;
subplot(2,1,2);
plot(global_position_compass_hdg2);
title('compass heading in degrees(irregular)');
hold off;
%% local position
local_position_pose1 = cell2mat(local_position_pose1.bMsg);
local_position_pose1_O = [local_position_pose1.Pose];
local_position_pose1_O = [local_position_pose1_O.Orientation];
local_position_pose1_O = [local_position_pose1_O(:).X;local_position_pose1_O(:).Y;local_position_pose1_O(:).Z;local_position_pose1_O(:).W];
local_position_pose2 = cell2mat(local_position_pose2.bMsg);
local_position_pose2_O = [local_position_pose2.Pose];
local_position_pose2_O = [local_position_pose2_O.Orientation];
local_position_pose2_O = [local_position_pose2_O(:).X;local_position_pose2_O(:).Y;local_position_pose2_O(:).Z;local_position_pose2_O(:).W];
% prikaz podataka
figure(2);
subplot(4,1,1);
plot(local_position_pose1_O(1,:));
hold on;
title('x');
plot(local_position_pose2_O(1,:));
legend('regular','irregular');
hold off;
subplot(4,1,2);
plot(local_position_pose1_O(2,:));
title('y');
hold on;
plot(local_position_pose2_O(2,:));
legend('regular','irregular');
hold off;
subplot(4,1,3);
plot(local_position_pose1_O(3,:));
title('z');
hold on;
plot(local_position_pose2_O(3,:));
legend('regular','irregular');
hold off;
subplot(4,1,4);
plot(local_position_pose1_O(4,:));
title('w');
hold on;
plot(local_position_pose2_O(4,:));
legend('regular','irregular');
hold off;
%==========================================================================
%% 3. KLASIFIKACIJA 
% NAPOMENA: Nece sva merenja sa IMU senzora biti iskoriscenja, ona koja se
% ne menjaju tokom vremena ce biti izbacena kasnije u kodu

% smestanja podataka iz foledra 03 u vektor obelezja X3 
IMU_static_pressure3 = vertcat(IMU_static_pressure3.bMsg);
IMU_static_pressure3 = cell2mat(IMU_static_pressure3);
IMU_static_pressure3 = [IMU_static_pressure3(:).FluidPressure];
IMU_diff_pressure3 = vertcat(IMU_diff_pressure3.bMsg);
IMU_diff_pressure3 = cell2mat(IMU_diff_pressure3);
IMU_diff_pressure3 = [IMU_diff_pressure3(:).FluidPressure];
IMU_data3 = vertcat(IMU_data3.bMsg);
IMU_data3 = cell2mat(IMU_data3);
IMU_data3_AVC = [IMU_data3(:).AngularVelocityCovariance];
IMU_data3_LAC = [IMU_data3(:).LinearAccelerationCovariance];
IMU_data3_OC = [IMU_data3(:).OrientationCovariance];
IMU_data3_O = [IMU_data3(:).Orientation];
IMU_data3_O = [IMU_data3_O(:).X;IMU_data3_O(:).Y;IMU_data3_O(:).Z;IMU_data3_O(:).W];
IMU_data3_LA = [IMU_data3(:).LinearAcceleration];
IMU_data3_LA = [IMU_data3_LA(:).X;IMU_data3_LA(:).Y;IMU_data3_LA(:).Z];
IMU_data3_AV = [IMU_data3(:).AngularVelocity];
IMU_data3_AV = [IMU_data3_AV(:).X;IMU_data3_AV(:).Y;IMU_data3_AV(:).Z];
IMU_mag3 = vertcat(IMU_mag3.bMsg);
IMU_mag3 = cell2mat(IMU_mag3);
IMU_mag3_MFC = [IMU_mag3(:).MagneticFieldCovariance];
IMU_mag3_MF = [IMU_mag3(:).MagneticField];
IMU_mag3_MF = [IMU_mag3_MF(:).X;IMU_mag3_MF(:).Y;IMU_mag3_MF(:).Z];
global_position_compass_hdg3 = cell2mat(vertcat(global_position_compass_hdg3.bMsg));
global_position_compass_hdg3 = [global_position_compass_hdg3(:).Data];
X3 = [IMU_static_pressure3(:,1:690);IMU_diff_pressure3(:,1:690);...
    IMU_data3_AVC(:,1:690);IMU_data3_LAC(:,1:690);IMU_data3_OC(:,1:690);...
    IMU_data3_O(:,1:690);IMU_data3_LA(:,1:690);IMU_data3_AV(:,1:690);...
    IMU_mag3_MFC(:,1:690);IMU_mag3_MF(:,1:690);global_position_compass_hdg3(:,1:690)];

% smestanja podataka iz foledra 04 u vektor obelezja X4 
IMU_static_pressure4 = vertcat(IMU_static_pressure4.bMsg);
IMU_static_pressure4 = cell2mat(IMU_static_pressure4);
IMU_static_pressure4 = [IMU_static_pressure4(:).FluidPressure];
IMU_diff_pressure4 = vertcat(IMU_diff_pressure4.bMsg);
IMU_diff_pressure4 = cell2mat(IMU_diff_pressure4);
IMU_diff_pressure4 = [IMU_diff_pressure4(:).FluidPressure];
IMU_data4 = vertcat(IMU_data4.bMsg);
IMU_data4 = cell2mat(IMU_data4);
IMU_data4_AVC = [IMU_data4(:).AngularVelocityCovariance];
IMU_data4_LAC = [IMU_data4(:).LinearAccelerationCovariance];
IMU_data4_OC = [IMU_data4(:).OrientationCovariance];
IMU_data4_O = [IMU_data4(:).Orientation];
IMU_data4_O = [IMU_data4_O(:).X;IMU_data4_O(:).Y;IMU_data4_O(:).Z;IMU_data4_O(:).W];
IMU_data4_LA = [IMU_data4(:).LinearAcceleration];
IMU_data4_LA = [IMU_data4_LA(:).X;IMU_data4_LA(:).Y;IMU_data4_LA(:).Z];
IMU_data4_AV = [IMU_data4(:).AngularVelocity];
IMU_data4_AV = [IMU_data4_AV(:).X;IMU_data4_AV(:).Y;IMU_data4_AV(:).Z];
IMU_mag4 = vertcat(IMU_mag4.bMsg);
IMU_mag4 = cell2mat(IMU_mag4);
IMU_mag4_MFC = [IMU_mag4(:).MagneticFieldCovariance];
IMU_mag4_MF = [IMU_mag4(:).MagneticField];
IMU_mag4_MF = [IMU_mag4_MF(:).X;IMU_mag4_MF(:).Y;IMU_mag4_MF(:).Z];
global_position_compass_hdg4 = cell2mat(vertcat(global_position_compass_hdg4.bMsg));
global_position_compass_hdg4 = [global_position_compass_hdg4(:).Data];
X4 = [IMU_static_pressure4(:,1:690);IMU_diff_pressure4(:,1:690);...
    IMU_data4_AVC(:,1:690);IMU_data4_LAC(:,1:690);IMU_data4_OC(:,1:690);...
    IMU_data4_O(:,1:690);IMU_data4_LA(:,1:690);IMU_data4_AV(:,1:690);...
    IMU_mag4_MFC(:,1:690);IMU_mag4_MF(:,1:690);global_position_compass_hdg4(:,1:690)];
%% Podela podataka i priprema za klasifikaciju
m3 = size(X3,2);
ind = randperm(m3);
X3 = X3(:, ind);
Y3 = zeros(1,m3);
m4 = size(X4,2);
ind = randperm(m4);
X4 = X4(:, ind);
Y4 = ones(1,m4);
% podela na train i test skup (odnos 90%:10%)
X3train = X3(:,1:0.9*m3);
Y3train = Y3(1,1:0.9*m3);
X4train = X4(:,1:0.9*m4);
Y4train = Y4(1,1:0.9*m4);
X3test = X3(:,0.9*m3+1:end);
Y3test = Y3(1,0.9*m3+1:end);
X4test = X4(:,0.9*m4+1:end);
Y4test = Y4(1,0.9*m4+1:end);
% sjedinjavanje skupova
Xtrain = [X3train X4train]';
Ytrain = [Y3train Y4train]';
Xtest = [X3test X4test]';
Ytest = [Y3test Y4test]';
ind = randperm(size(Xtrain,1));
Xtrain = Xtrain(ind,:);
Ytrain = Ytrain(ind,1);
ind = randperm(size(Xtest,1));
Xtest = Xtest(ind,:);
Ytest = Ytest(ind,1);

%% Treniranje logisticke regresije
clc;
[X, X_test] = FeatureScaling(Xtrain, Xtest); % takodje odbacuje obelezja sa std=0;
% Dodavanje polinomijalnih obelezja
X = AddPoliFeatures(X); % ovo dodaje i kolonu jedinica za bias
% Inicijalizacija parametara
y = Ytrain;
initial_theta = zeros(size(X, 2), 1);
lambda = 0;
options = optimset('GradObj', 'on', 'MaxIter', 400); % Postavljanje opcija
% Treniranje
[theta, J, exit_flag] = fminunc(@(t)(CostFuncReg(t, X, y, lambda)), initial_theta, options); 
% Racunanje preciznosti
p_train = Predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p_train == y)) * 100);
X_test = AddPoliFeatures(X_test);
p_test = Predict(theta, X_test);
fprintf('Test Accuracy: %f\n', mean(double(p_test == Ytest)) * 100);
%% Cuvanje parametara
theta_trained = theta;
save('theta.mat','theta_trained');


