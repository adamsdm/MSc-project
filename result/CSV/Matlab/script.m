
%% Read input data
clc;
close all;
clear all;

load("stuff.mat");

%% Total execution time
noBodies = cl_data(:,1);
cl_tTot = cl_data(:,6);
cu_tTot = cu_data(:,6);
dc_tTot = dc_data(:,6);
seq_tTot = seq_data(:,6);

figure(1)
plot(noBodies,[cl_tTot cu_tTot dc_tTot seq_tTot]);

% Adjust the axis limits
axis([1024 10*1024 0 160])

title('Total execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('OpenCL', 'CUDA', 'DirectCompute', 'Sequential');


saveas(gcf,'Plots/TotalExecutionTime.png')

clear cl_tTot cu_tTot dc_tTot seq_tTot noBodies;


%% GPU Step execution time

noBodies = cl_data(:,1);
cl_tTot = cl_data(:,5);
cu_tTot = cu_data(:,5);
dc_tTot = dc_data(:,5);

figure(2)
plot(noBodies,[cl_tTot cu_tTot dc_tTot]);



title('GPU Step execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('OpenCL', 'CUDA', 'DirectCompute');

saveas(gcf,'Plots/GPUStepExecutionTime.png')
clear cl_tTot cu_tTot dc_tTot seq_tTot noBodies;


%% CUDA Bar Chart

data = cu_data;

% Extract data
noBodies = data(:,1);
tBuildTree = data(:,2);
tCalcTreeCOM = data(:,3);
tFlattenTree = data(:,4);
tStep = data(:,5);
tTot = data(:,6);

figure(3)
bar(noBodies, [tBuildTree tCalcTreeCOM tFlattenTree tStep], 0.5, 'stack')

% Adjust the axis limits
axis([0 11*1024 0 60])


% Add title and axis labels
title('CUDA execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('Build Tree', 'Calc Tree COM', 'Flatten Tree', 'CU Step');

saveas(gcf,'Plots/CUDABarChart.png');
clear data noBodies tBuildTree tCalcTreeCOM tFlattenTree tStep tTot;

%% OpenCL Bar Chart

data = cl_data;

% Extract data
noBodies = data(:,1);
tBuildTree = data(:,2);
tCalcTreeCOM = data(:,3);
tFlattenTree = data(:,4);
tStep = data(:,5);
tTot = data(:,6);

figure(4)
bar(noBodies, [tBuildTree tCalcTreeCOM tFlattenTree tStep], 0.5, 'stack')

% Adjust the axis limits
axis([0 11*1024 0 40])


% Add title and axis labels
title('OpenCL execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('Build Tree', 'Calc Tree COM', 'Flatten Tree', 'CL Step');
saveas(gcf,'Plots/OpenCLBarChart.png');
clear data noBodies tBuildTree tCalcTreeCOM tFlattenTree tStep tTot;

%% DirectCompute Bar Chart

data = dc_data;

% Extract data
noBodies = data(:,1);
tBuildTree = data(:,2);
tCalcTreeCOM = data(:,3);
tFlattenTree = data(:,4);
tStep = data(:,5);
tTot = data(:,6);

figure(5)
bar(noBodies, [tBuildTree tCalcTreeCOM tFlattenTree tStep], 0.5, 'stack')


% Adjust the axis limits
axis([0 11*1024 0 45])


% Add title and axis labels
title('DirectCompute execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('Build Tree', 'Calc Tree COM', 'Flatten Tree', 'DC Step');
saveas(gcf,'Plots/DirectComputeBarChart.png');
clear data noBodies tBuildTree tCalcTreeCOM tFlattenTree tStep tTot;


%% Sequential Bar Chart

data = seq_data;

% Extract data
noBodies = data(:,1);
tBuildTree = data(:,2);
tCalcTreeCOM = data(:,3);
tFlattenTree = data(:,4);
tStep = data(:,5);
tTot = data(:,6);

figure(6)
bar(noBodies, [tBuildTree tCalcTreeCOM tFlattenTree tStep], 0.5, 'stack')


% Adjust the axis limits
axis([0 11*1024 0 160])


% Add title and axis labels
title('Sequential execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('Build Tree', 'Calc Tree COM', 'Flatten Tree', 'Seq Step');
saveas(gcf,'Plots/SequentialBarChart.png');
clear data noBodies tBuildTree tCalcTreeCOM tFlattenTree tStep tTot;

