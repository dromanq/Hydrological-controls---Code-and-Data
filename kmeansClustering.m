%%%%% THIS SCRIPT ALLOWS TO INVESTIGATE A SYNTHETIC DATASET CONTAINING RAINFALL, SOIL
%%%%% MOISTURE AND AQUIFER WATER LEVEL INFORMATION USING THE K-MEANS
%%%%% CLUSTERING ALGORITHM

%% Clear workspace and load data. Please save the dataset "SyntheticData.mat" in the same folder of this script
clear; clc
close all

load('SyntheticData.mat')

%% Normalize and standardize the data to be clustered

DS2_H = Dstor2./H;

mH = mean(H); sdH = std(H);
mDS2_H = mean(DS2_H); stdDS2_H = std(DS2_H);
mhf = mean(hfalda_p); sdhf = std(hfalda_p);
mtheta100 = mean(theta_100_p); sdtheta100 = std(theta_100_p);

R4 = (hfalda_p - mhf)./(sdhf);
R5 = (theta_100_p - mtheta100)./(sdtheta100);
R6 = (DS2_H - mDS2_H)./ (stdDS2_H);

kR654 = kmeans([R5,R4,R6],4);

%% Represent the data in a scatterplot coloring it according to the cluster labes

figure
scatter3(theta_100_p(kR654 == 1),hfalda_p(kR654 == 1),H(kR654 == 1),15,[0.8,0.3,0.1],'filled')
hold on
scatter3(theta_100_p(kR654 == 2),hfalda_p(kR654 == 2),H(kR654 == 2),15,[0.9,0.7,0.1],'filled')
scatter3(theta_100_p(kR654 == 3),hfalda_p(kR654 == 3),H(kR654 == 3),15,[0,0.4,0.7],'filled')
scatter3(theta_100_p(kR654 == 4),hfalda_p(kR654 == 4),H(kR654 == 4),15,[0.6,0.1,0.2],'filled')
xlabel('\theta (-)')
ylabel('h_{a} (mm)')
zlabel('H (mm)')
set(gca,'XScale','log')
set(gca,'YScale','log')
set(gca,'ZScale','log')
set(gca,'FontSize',24)
legend('Cluster 1','Cluster 2','Cluster 3', 'Cluster 4')

%% Evaluate cluster performance with the Silhouette metric

E654 = evalclusters([R5,R4,R6],'kmeans','silhouette','klist',1:7);

figure
plot(E654)
title('Cluster evaluation for: R6,R5,R4')
set(gca,'FontSize',24)

