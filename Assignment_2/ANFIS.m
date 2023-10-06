%% Plot accuracies

accu_init_list = [];
accu_final_list = [];

for i=1:10
    [accuracy_init, accuracy_final] = ANFIS1();
    accu_init_list = [accu_init_list, accuracy_init];
    accu_final_list = [accu_final_list, accuracy_final];
end

figure;
plot(1:10, accu_init_list, 1:10, accu_final_list)
xlabel('Iteration')
ylabel('Accuracy')
legend('Initial fuzzy inference system', 'Tuned fuzzy inference system - ANFIS')
title('FIS and tuned FIS comparison')

% Save the figure as a PNG image
saveas(gcf, 'FISvsANFIS.png');

%% Save predictions to file

% Combine the vectors
combinedVectorFIS = [class_report_init.GroundTruth-1, Y_pred_init_labels];
combinedVectorANFIS = [class_report_final.GroundTruth-1, Y_pred_final_labels];

fileID = fopen('classification_fis.csv','w');
fprintf(fileID,'%6s %12s\n','Ground Truth','Prediction');
fprintf(fileID,'%6d %12d\n', combinedVectorFIS);
fclose(fileID);

fileID = fopen('classification_anfis.csv','w');
fprintf(fileID,'%6s %12s\n','Ground Truth','Prediction');
fprintf(fileID,'%6d %12d\n', combinedVectorANFIS);
fclose(fileID);

%% Define function

function [accuracy_init, accuracy_final] = ANFIS1()
    
    %% Load data and create train-test sets
    
    df = csvread('wine.data');
    Y = df(:,1) - 1;
    X = df(:,2:end);
    
    train_test_partition = cvpartition(Y,'Holdout',0.8,'Stratify',true);
    train_idx = training(train_test_partition);
    test_idx = test(train_test_partition);
    X_train = X(train_idx,:);
    X_test = X(test_idx,:);
    Y_train = Y(train_idx,:);
    Y_test = Y(test_idx,:);

    %% Train initial 0-vs-all model
    
    Y_train_0_vs_all = Y_train;
    Y_train_0_vs_all(Y_train_0_vs_all==0) = -1;
    Y_train_0_vs_all(Y_train_0_vs_all~=-1) = 0;
    Y_train_0_vs_all(Y_train_0_vs_all==-1) = 1;
    
    opt = genfisOptions('FCMClustering','FISType','sugeno');
    opt.NumClusters = 3;
    ts_model_0_vs_all = genfis(X_train,Y_train_0_vs_all,opt);
    
    %% Train initial 1-vs-all model
    
    Y_train_1_vs_all = Y_train;
    Y_train_1_vs_all(Y_train_1_vs_all==1) = -1;
    Y_train_1_vs_all(Y_train_1_vs_all~=-1) = 0;
    Y_train_1_vs_all(Y_train_1_vs_all==-1) = 1;
    
    opt = genfisOptions('FCMClustering','FISType','sugeno');
    opt.NumClusters = 3;
    ts_model_1_vs_all = genfis(X_train,Y_train_1_vs_all,opt);
    
    %% Train initial 2-vs-all model
    
    Y_train_2_vs_all = Y_train;
    Y_train_2_vs_all(Y_train_2_vs_all==2) = -1;
    Y_train_2_vs_all(Y_train_2_vs_all~=-1) = 0;
    Y_train_2_vs_all(Y_train_2_vs_all==-1) = 1;
    
    opt = genfisOptions('FCMClustering','FISType','sugeno');
    opt.NumClusters = 3;
    ts_model_2_vs_all = genfis(X_train,Y_train_2_vs_all,opt);
    
    %% Check initial performance on test set
    
    Y_pred_init_0_vs_all = evalfis(ts_model_0_vs_all, X_test);
    Y_pred_init_0_vs_all(Y_pred_init_0_vs_all<0) = 0;
    Y_pred_init_0_vs_all(Y_pred_init_0_vs_all>1) = 1;
    
    Y_pred_init_1_vs_all = evalfis(ts_model_1_vs_all, X_test);
    Y_pred_init_1_vs_all(Y_pred_init_1_vs_all<0) = 0;
    Y_pred_init_1_vs_all(Y_pred_init_1_vs_all>1) = 1;
    
    Y_pred_init_2_vs_all = evalfis(ts_model_2_vs_all, X_test);
    Y_pred_init_2_vs_all(Y_pred_init_2_vs_all<0) = 0;
    Y_pred_init_2_vs_all(Y_pred_init_2_vs_all>1) = 1;
    
    Y_pred_init = zeros(size(Y_test,1),3);
    Y_pred_init(:,1) = Y_pred_init_0_vs_all + (1-Y_pred_init_1_vs_all) + (1-Y_pred_init_2_vs_all);
    Y_pred_init(:,2) = Y_pred_init_1_vs_all + (1-Y_pred_init_0_vs_all) + (1-Y_pred_init_2_vs_all);
    Y_pred_init(:,3) = Y_pred_init_2_vs_all + (1-Y_pred_init_0_vs_all) + (1-Y_pred_init_1_vs_all);
    Y_pred_init = Y_pred_init./sum(Y_pred_init,2);
    [~,Y_pred_init_labels] = max(Y_pred_init,[],2);
    Y_pred_init_labels = Y_pred_init_labels - 1;
    
    class_report_init = classperf(Y_test, Y_pred_init_labels);
    fprintf('Initial Accuracy: %4.3f \n', class_report_init.CorrectRate);
    
    %% Tune initial 0-vs-all model using ANFIS
    
    [in,out,rule] = getTunableSettings(ts_model_0_vs_all);
    anfis_model_0_vs_all = tunefis(ts_model_0_vs_all,[in;out],X_train,Y_train_0_vs_all,tunefisOptions("Method","anfis"));
    
    %% Tune initial 1-vs-all model using ANFIS
    
    [in,out,rule] = getTunableSettings(ts_model_1_vs_all);
    anfis_model_1_vs_all = tunefis(ts_model_1_vs_all,[in;out],X_train,Y_train_1_vs_all,tunefisOptions("Method","anfis"));
    
    %% Tune initial 2-vs-all model using ANFIS
    
    [in,out,rule] = getTunableSettings(ts_model_2_vs_all);
    anfis_model_2_vs_all = tunefis(ts_model_2_vs_all,[in;out],X_train,Y_train_2_vs_all,tunefisOptions("Method","anfis"));
    
    %% Check ANFIS tuned model performance
    
    Y_pred_final_0_vs_all = evalfis(anfis_model_0_vs_all, X_test);
    Y_pred_final_0_vs_all(Y_pred_final_0_vs_all<0) = 0;
    Y_pred_final_0_vs_all(Y_pred_final_0_vs_all>1) = 1;
    
    Y_pred_final_1_vs_all = evalfis(anfis_model_1_vs_all, X_test);
    Y_pred_final_1_vs_all(Y_pred_final_1_vs_all<0) = 0;
    Y_pred_final_1_vs_all(Y_pred_final_1_vs_all>1) = 1;
    
    Y_pred_final_2_vs_all = evalfis(anfis_model_2_vs_all, X_test);
    Y_pred_final_2_vs_all(Y_pred_final_2_vs_all<0) = 0;
    Y_pred_final_2_vs_all(Y_pred_final_2_vs_all>1) = 1;
    
    Y_pred_final = zeros(size(Y_test,1),3);
    Y_pred_final(:,1) = Y_pred_final_0_vs_all + (1-Y_pred_final_1_vs_all) + (1-Y_pred_final_2_vs_all);
    Y_pred_final(:,2) = Y_pred_final_1_vs_all + (1-Y_pred_final_0_vs_all) + (1-Y_pred_final_2_vs_all);
    Y_pred_final(:,3) = Y_pred_final_2_vs_all + (1-Y_pred_final_0_vs_all) + (1-Y_pred_final_1_vs_all);
    Y_pred_final = Y_pred_final./sum(Y_pred_final,2);
    [~,Y_pred_final_labels] = max(Y_pred_final,[],2);
    Y_pred_final_labels = Y_pred_final_labels - 1;
    
    class_report_final = classperf(Y_test, Y_pred_final_labels);
    fprintf('Final Accuracy: %4.3f \n', class_report_final.CorrectRate);
    
    accuracy_init = class_report_init.CorrectRate;
    accuracy_final = class_report_final.CorrectRate;
    
    % USE VALIDATION AND PRESENT THE OVERFIT GRAPH (ERROR IN TRAIN VS ERROR IN
    % VALID)
    
    %% Perform ANFIS using different functions to produce convergence plot
    
    % Create the dataset 
    df = csvread('wine.data');
    Y = df(:,1) - 1;
    X = df(:,2:end);
    
    % Define the split sizes 
    trainSize=0.7*size(df, 1); 
    valSize=0.2*size(df, 1); 
    testSize=0.1*size(df, 1); 
    
    % Create the training data 
    cv=cvpartition(size(df, 1),'HoldOut',0.3); 
    idx=cv.test; 
    dataTrain = df(~idx,:); 
    dataValTest  = df(idx,:); 
    
    % Create validation and test data 
    cv=cvpartition(53,'HoldOut',0.5); 
    idx=cv.test; 
    dataVal = df(~idx,:); 
    dataTest  = df(idx,:); 
    
    X_train = dataTrain(:,2:14);
    Y_train = dataTrain(:,1);
    X_test = dataTest(:,2:14);
    Y_test = dataTest(:,1);
    X_valid = dataVal(:,2:14);
    Y_valid = dataVal(:,1);
end
