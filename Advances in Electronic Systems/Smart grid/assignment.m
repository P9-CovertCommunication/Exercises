%% Exercise 1
clear all; close all; clc;


num_smart_meters = 3;

lambda=[1, 0.6, 0.8];               %Lambda values for all three meters, organized the way they are accessed
delays=[1 1 .5 .5 .1 .1];     %Delay for all three meters, following <d1,d2><d3,d4><d5,d6> 


%Error check
if not(length(lambda) == num_smart_meters) && not(length(delays) == num_smart_meters)
    error("Mismatch in lambda or delay list length and number of smart meters")
end

%Get all permuatations of smart meter orders
permutations = perms(1:num_smart_meters);
num_permutations = size(permutations, 1);

%Arrays for storing all results
mmPr_results = zeros(num_permutations,num_smart_meters);
mean_results = zeros(num_permutations,1);
var_results = zeros(num_permutations,1);

for i = 1:num_permutations
    current_perm = permutations(i,:);

    mmPr = zeros(1,num_smart_meters); %Array to store mmPr of each smart meter

    for k = 1:num_smart_meters
        sm_num = current_perm(k);
        subDelay = delays(2:(sm_num*2));
        mmPr(k) = getmmPr(subDelay, lambda(sm_num));
    end

    %Add result to array
    mmPr_results(i,:) = mmPr;
    mean_results(i) = mean(mmPr);
    var_results(i) = var(mmPr);
end

%Print results
disp('mmPr Results for each permutation:');
disp(mmPr_results);
disp('Mean mmPr for each permutation:');
disp(mean_results);
disp('Variance of mmPr for each permutation:');
disp(var_results);

%Find best permuataion
[min_mean, min_mean_arg] = min(mean_results);
[min_var, min_var_arg] = min(var_results);

disp("Best permutation for min variance")
disp(permutations(min_var_arg,:))
fprintf("min mmPr variance: %d\n\n",min_var)

disp("Best permutation for min mean")
disp(permutations(min_mean_arg,:))
fprintf("min mean mmPr: %d\n\n",min_mean)



%% Exercise 2

clear all; close all; clc;

num_smart_meters = 10;

%Random uniform lambdas in the interval 0.5-1
lambda_bounds = [0.5, 1];
lambda = lambda_bounds(1) + (lambda_bounds(2)-lambda_bounds(1)).*rand(num_smart_meters,1);

%random uniform delays in the interval 0.1-2
delay_bounds = [0.05, 1];
delays= delay_bounds(1)+(delay_bounds(2)-delay_bounds(1)).*rand(num_smart_meters*2,1); %Independent random delay pairs

% %Identical delay pairs
% delays= delay_bounds(1)+(delay_bounds(2)-delay_bounds(1)).*rand(num_smart_meters,1);
% delays = reshape(repmat(delays, 1, 2)', [], 1); 

%Error check
if not(length(lambda) == num_smart_meters) || not(length(delays)==num_smart_meters*2)
    error("Mismatch in lambda or delay list length and number of smart meters")
end

%Get all permuatations of smart meter orders
permutations = perms(1:num_smart_meters);
num_permutations = size(permutations, 1);

%Arrays for storing all results
mmPr_results = zeros(num_permutations,num_smart_meters);
mean_results = zeros(num_permutations,1);
var_results = zeros(num_permutations,1);

for i = 1:num_permutations
    current_perm = permutations(i);

    mmPr = zeros(1,num_smart_meters); %Array to store mmPr of each smart meter

    for k = 1:num_smart_meters
        sm_num = current_perm(k);
        subDelay = delays(2:(k*2));
        mmPr(k) = getmmPr(subDelay, lambda(sm_num));
    end

    %Add result to array
    mmPr_results(i,:) = mmPr;
    mean_results(i) = mean(mmPr);
    var_results(i) = var(mmPr);
    if mod(i/num_permutations*100,1) == 0
        clc
        fprintf("Percentage done %d \n\n",floor(i/num_permutations*100))
    end
end

% %Print results (results are probably too large to be useful 
% disp('mmPr Results for each permutation:');
% disp(mmPr_results);
% disp('Mean mmPr for each permutation:');
% disp(mean_results);
% disp('Variance of mmPr for each permutation:');
% disp(var_results);

%Find best permuataion
[min_mean, min_mean_arg] = min(mean_results);
[min_var, min_var_arg] = min(var_results);

disp("Best permutation for min variance")
disp(permutations(min_var_arg,:))
fprintf("min mmPr variance: %d\n\n",min_var)

disp("Best permutation for min mean")
disp(permutations(min_mean_arg,:))
fprintf("min mean mmPr: %d\n\n",min_mean)


%% Discussion
% How can we approach the complexity of 10 or more smart meters?
% Better search algorithm for optimal permutation. Dynamic programming,
% machine learning, combinatorial optimization problem methods like the
% travelling salesman problem
%
%
%
%
%
%
%% Function definitions
%NB! Any reordering of the access, e.g. lambda=[.6,1,.8] will also lead to
%a restructure of the delays - in this case: <d5,d6><d1,d2><d3,d4>
%For the first excercise, where all delays are equal, it has no impact, but
%with different delays for each SM it has an impact!!

%Basically, the question boils down to - find a smart way to organize the
%access order such that e.g. mean(mmPr) is minimum, and/or var(mmPr) is
%minimum


%Returns the mmPr given delay and event metrics
function [mmPr] = getmmPr(subDelay,lambda)

%First use this function to get vector pair <pd,Bd>
[pd,Bd]=getVectorMatriceDelayPairs(subDelay);

%Input vector/matrix pair for the delay <pd,Bd> and an average time
%interval between events [sec]. Equation here assumes exp. random time intervals.
kron_sum=inv(lambda*eye(size(Bd,1))+Bd)*ones(size(Bd,1),1);
mmPr=1-(pd*Bd)*kron_sum;
end

%Construct the necessary matrices for the calculation in getmmPr()
function [p,B]=getVectorMatriceDelayPairs(delays)
%Input: vector of delays from measurement send. All delays are exp. random delays in [sec].
p=zeros(1,length(delays));
p(1)=1;

%This constructs a Matrix Exponential description for an Erlang distribution
B0=diag(1./delays);                         %we need to convert to rates and not time
B1=[zeros(length(delays),1), B0]; %Add a row of zeros
B1(:,size(B1,2))=[];                        %remove the last part of B0
B=B0-B1;                                       %Construct the final matrix
end