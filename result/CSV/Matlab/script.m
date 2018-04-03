
%% Read input data
clc;
close all;
clear all;

load("all_data.mat");

%% Total execution time
noBodies = cl_data(:,1);
cl_tTot = cl_data(:,6);
cu_tTot = cu_data(:,6);
dc_tTot = dc_data(:,6);
seq_tTot = seq_data(:,6);


%%

figure(1)
hold on;
plot(noBodies, cl_tTot);
plot(noBodies, cu_tTot);
plot(noBodies, dc_tTot);
plot(noBodies, seq_tTot);

% Adjust the axis limits
axis([1024 20*1024 0 400])

title('Total execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('OpenCL', 'CUDA', 'DirectCompute', 'Sequential');

%%
saveas(gcf,'Plots/TotalExecutionTime.png');

%% GPU Step execution time

noBodies = cl_data(:,1);
cl_tTot = cl_data(:,5);
cu_tTot = cu_data(:,5);
dc_tTot = dc_data(:,5);

figure(2)
plot(noBodies,[cl_tTot cu_tTot dc_tTot]);

% Adjust the axis limits
axis([1024 20*1024 0 50])


title('GPU Step execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('OpenCL', 'CUDA', 'DirectCompute');

%%
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
axis([0 21*1024 0 130])


% Add title and axis labels
title('CUDA execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('Build Tree', 'Calc Tree COM', 'Flatten Tree', 'CU Step');


%%
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
axis([0 21*1024 0 90])


% Add title and axis labels
title('OpenCL execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('Build Tree', 'Calc Tree COM', 'Flatten Tree', 'CL Step');

%%

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
axis([0 21*1024 0 95])


% Add title and axis labels
title('DirectCompute execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('Build Tree', 'Calc Tree COM', 'Flatten Tree', 'DC Step');

%%
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
axis([0 21*1024 0 390])


% Add title and axis labels
title('Sequential execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('Build Tree', 'Calc Tree COM', 'Flatten Tree', 'Seq Step');

%%
saveas(gcf,'Plots/SequentialBarChart.png');
clear data noBodies tBuildTree tCalcTreeCOM tFlattenTree tStep tTot;



%%
noBodies = cu_data(:,1);
tTot_struct = cu_data(:,6);
tTot_class = cu_data_class(:,6);

figure(7);
hold on;
plot(noBodies, tTot_struct);
plot(noBodies, tTot_class);
hold off;

% Adjust the axis limits
axis([1024 21*1024 0 120])


% Add title and axis labels
title('CUDA struct and class buffer execution time')
xlabel('Number of bodies')
ylabel('Execution time (ms)')

legend('Struct', 'Class');

%%
saveas(gcf,'Plots/CUDAStructVSClass.png');
clear data noBodies tBuildTree tCalcTreeCOM tFlattenTree tStep tTot;