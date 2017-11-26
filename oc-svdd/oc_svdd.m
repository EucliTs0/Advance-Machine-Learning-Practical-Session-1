clear all
clc
close all


% load breast cancer data
data = csvread('wdbc_data.csv');
labels = data(:, 31);
data(:, 31) = [];
data = (data - min(data)) ./ (max(data) - min(data));
%data = zscore(data);
labels(labels == 1) = -1;
labels(labels == 2) = 1;

%malignant = data(labels == -1, :);
%benign = data(labels == 1, :);
%test_labels = -ones(size(malignant, 1), 1);
train_range = [5 10 20 50 80 100 120 150 180 200];
iterations = 10;

randn('state',0) ;
rand('state',0) ;

for r = 1:length(train_range)
    for iter = 1:iterations
        malignant = data(labels == -1, :);
        benign = data(labels == 1, :);
        test_labels = -ones(size(malignant, 1), 1);
        k = randperm(length(benign), train_range(r));
        remaining_benign = benign;
        remaining_benign(k, :) = [];
        benign = benign(k, :);
        train_data = benign;

        malignant = cat(1, malignant, remaining_benign);
        test_data = malignant;
        train_labels = ones(size(train_data, 1), 1);
        test_labels = cat(1, test_labels, ones(size(remaining_benign, 1), 1));
        %%

        nu = [0.001 0.005 0.01 0.03 0.05 0.1 0.2];
        
        %ker = [2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3 2^4 2^5 2^6 2^7];
        ker2 = [[4 1];[3 1];[2 1];[1 1];];


        %%
        best_perf = 0;

        for i = 1:length(nu)
            %for j = 1:length(ker)
            for j = 1:size(ker2, 1)
                C = 1 ./ (nu(i)*length(train_data));
                SD = simpleData(train_data',train_labels);
                %SD.kernelType = 'rbf';
                SD.kernelType = 'poly';
%                 SD.kernelParam = ker(j);
                SD.kernelParam = ker2(j, :);

                SM = simpleModelSVDD(SD,1/(nu(i)*length(train_data)));
                SM.train;
                %ypred = SM.b - ones(size(testData, 2), 1) + 2*(SM.alpha'*testData([SM.Iw,SM.Iu],:))';
                %perfV = 100*length(find(sign(ypred)==test_labels))/length(test_labels);

                [ypred,perfV] = SM.test(test_data',test_labels);

                if perfV > best_perf
                    best_perf = perfV;
                    %best_k = ker(j);
                    best_p = ker2(j, :);
                    best_nu = nu(i);
                    best_C = C;
                    best_ypred = ypred;
                end
            end
        end
        acc{r}{iter} = best_perf;
        perf_(iter) = best_perf;
        nu_best(iter) = best_nu;
        %gamma_best{r}{iter} = best_k;
        p_best(iter, :) = best_p;
        C_best(iter) = best_C;
        preds{r}{iter} = best_ypred;
        
    end
    avg_perf(r) = sum(perf_) / length(perf_);
end
%%
figure(1)
loglog(train_range, avg_perf);
axis([0 250 0 100])
xlabel('Training size, logscale');
ylabel('Average Performance');
grid on


        


    
