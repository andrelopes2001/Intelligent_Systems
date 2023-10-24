%% Load data

tic

train_data = readtable('dataframes\train_df.csv');
test_data = readtable('dataframes\test_df.csv');

X_train = table2array(train_data(1:end, 1:end-1));
Y_train = train_data.Type; 

X_test = table2array(test_data(1:end, 1:end-1));
Y_test = test_data.Type; 

% Define a mapping from category strings to numerical values
category_map = containers.Map({'F', 'N', 'Q', 'S', 'V'}, {0, 1, 2, 3, 4});

% Use the mapping to convert the categorical variable to numerical values
Y_train = cellfun(@(x) category_map(x), Y_train);
Y_test = cellfun(@(x) category_map(x), Y_test);

recall_list_init = [];
f1score_list_init = [];
mcc_list_init = [];
kappa_list_init = [];
cm_list_init = [];

opt = genfisOptions('FCMClustering', 'FISType', 'sugeno');
opt.NumClusters = 5;
opt.Verbose = 0;

% Train 0-vs-all model
Y_train_0 = Y_train;
Y_train_0(Y_train_0==0) = -1;
Y_train_0(Y_train_0~=-1) = 0;
Y_train_0(Y_train_0==-1) = 1;
model0 = genfis(X_train, Y_train_0, opt);

% Train 1-vs-all model
Y_train_1 = Y_train;
Y_train_1(Y_train_1==1) = -1;
Y_train_1(Y_train_1~=-1) = 0;
Y_train_1(Y_train_1==-1) = 1;
model1 = genfis(X_train, Y_train_1, opt);

% Train 2-vs-all model
Y_train_2 = Y_train;
Y_train_2(Y_train_2==2) = -1;
Y_train_2(Y_train_2~=-1) = 0;
Y_train_2(Y_train_2==-1) = 1;
model2 = genfis(X_train, Y_train_2, opt);

% Train 3-vs-all model
Y_train_3 = Y_train;
Y_train_3(Y_train_3==3) = -1;
Y_train_3(Y_train_3~=-1) = 0;
Y_train_3(Y_train_3==-1) = 1;
model3 = genfis(X_train, Y_train_3, opt);

% Train 4-vs-all model
Y_train_4 = Y_train;
Y_train_4(Y_train_4==2) = -1;
Y_train_4(Y_train_4~=-1) = 0;
Y_train_4(Y_train_4==-1) = 1;
model4 = genfis(X_train, Y_train_4, opt);

% Evaluate the FIS on the test data
Y_pred_0 = evalfis(model0, X_test);
Y_pred_0(Y_pred_0<0) = 0;
Y_pred_0(Y_pred_0>1) = 1;

Y_pred_1 = evalfis(model1, X_test);
Y_pred_1(Y_pred_1<0) = 0;
Y_pred_1(Y_pred_1>1) = 1;

Y_pred_2 = evalfis(model2, X_test);
Y_pred_2(Y_pred_2<0) = 0;
Y_pred_2(Y_pred_2>1) = 1;

Y_pred_3 = evalfis(model3, X_test);
Y_pred_3(Y_pred_3<0) = 0;
Y_pred_3(Y_pred_3>1) = 1;

Y_pred_4 = evalfis(model4, X_test);
Y_pred_4(Y_pred_4<0) = 0;
Y_pred_4(Y_pred_4>1) = 1;

Y_pred = zeros(size(Y_test,1),5);
Y_pred(:,1) = Y_pred_0 + (1-Y_pred_1) + (1-Y_pred_2) + (1-Y_pred_3) + (1-Y_pred_4);
Y_pred(:,2) = Y_pred_1 + (1-Y_pred_0) + (1-Y_pred_2) + (1-Y_pred_3) + (1-Y_pred_4);
Y_pred(:,3) = Y_pred_2 + (1-Y_pred_0) + (1-Y_pred_1) + (1-Y_pred_3) + (1-Y_pred_4);
Y_pred(:,4) = Y_pred_3 + (1-Y_pred_0) + (1-Y_pred_1) + (1-Y_pred_2) + (1-Y_pred_4);
Y_pred(:,5) = Y_pred_4 + (1-Y_pred_0) + (1-Y_pred_1) + (1-Y_pred_2) + (1-Y_pred_3);

Y_pred = Y_pred ./ sum(Y_pred, 2);
[~, Y_pred_labels] = max(Y_pred, [], 2);
Y_pred_labels = Y_pred_labels - 1;

class_report_init = classperf(Y_test, Y_pred_labels);

sensitivity = class_report_init.Sensitivity;
specificity = class_report_init.Specificity;
ppv = class_report_init.PositivePredictiveValue;

% Calculate Recall
recall = sensitivity;

% Calculate F1 Score
f1Score = 2 * (sensitivity * ppv)/(sensitivity + ppv);

% Calculate cm
cm = confusionmat(Y_test,Y_pred_labels);

% Calculate Cohen's Kappa
num_class = 5;
observed_agreement = sum(diag(cm));
row_totals = sum(cm, 2);
col_totals = sum(cm, 1);
expected_agreement = 0;
for i = 1:num_class
    expected_agreement = expected_agreement + (row_totals(i)*col_totals(i)) / sum(row_totals);
end
kappa = (observed_agreement - expected_agreement) / (sum(row_totals) - expected_agreement);
 
% Calculate Matthews Correlation Coefficient
mcc = (sensitivity * ppv - sqrt((1 - sensitivity) * (1 - specificity) * sensitivity * (1 - ppv))) / ...
      sqrt((sensitivity + specificity - sensitivity * specificity) * (sensitivity + ppv - sensitivity * ppv));

% Store score metrics
f1score_list_init = [f1score_list_init, f1Score];
recall_list_init = [recall_list_init, recall];
mcc_list_init = [mcc_list_init, mcc];
kappa_list_init = [kappa_list_init, kappa];
cm_list_init = [cm_list_init, cm];

% % Check membership functions sigma values
% model0 = verifySigma(model0);
% model1 = verifySigma(model1);
% model2 = verifySigma(model2);
% model3 = verifySigma(model3);
% model4 = verifySigma(model4);

% % Tune using ANFIS
% [in, out, ~] = getTunableSettings(model0);
% anfis0 = tunefis(model0,[in;out],X_train,Y_train_0,tunefisOptions("Method","anfis"));
% 
% [in, out, ~] = getTunableSettings(model1);
% anfis1 = tunefis(model1,[in;out],X_train,Y_train_1,tunefisOptions("Method","anfis"));
% 
% [in, out, ~] = getTunableSettings(model2);
% anfis2 = tunefis(model2,[in;out],X_train,Y_train_2,tunefisOptions("Method","anfis"));
% 
% [in, out, ~] = getTunableSettings(model3);
% anfis3 = tunefis(model3,[in;out],X_train,Y_train_3,tunefisOptions("Method","anfis"));
% 
% [in, out, ~] = getTunableSettings(model4);
% anfis4 = tunefis(model4,[in;out],X_train,Y_train_4,tunefisOptions("Method","anfis"));
% 
% % Evaluate ANFIS on the test data
% Y_pred_0 = evalfis(anfis0, X_test);
% Y_pred_0(Y_pred_0<0) = 0;
% Y_pred_0(Y_pred_0>1) = 1;
% 
% Y_pred_1 = evalfis(anfis1, X_test);
% Y_pred_1(Y_pred_1<0) = 0;
% Y_pred_1(Y_pred_1>1) = 1;
% 
% Y_pred_2 = evalfis(anfis2, X_test);
% Y_pred_2(Y_pred_2<0) = 0;
% Y_pred_2(Y_pred_2>1) = 1;
% 
% Y_pred_3 = evalfis(anfis3, X_test);
% Y_pred_3(Y_pred_3<0) = 0;
% Y_pred_3(Y_pred_3>1) = 1;
% 
% Y_pred_4 = evalfis(anfis4, X_test);
% Y_pred_4(Y_pred_4<0) = 0;
% Y_pred_4(Y_pred_4>1) = 1;
% 
% Y_pred = zeros(size(Y_test,1),5);
% Y_pred(:,1) = Y_pred_0 + (1-Y_pred_1) + (1-Y_pred_2) + (1-Y_pred_3) + (1-Y_pred_4);
% Y_pred(:,2) = Y_pred_1 + (1-Y_pred_0) + (1-Y_pred_2) + (1-Y_pred_3) + (1-Y_pred_4);
% Y_pred(:,3) = Y_pred_2 + (1-Y_pred_0) + (1-Y_pred_1) + (1-Y_pred_3) + (1-Y_pred_4);
% Y_pred(:,2) = Y_pred_3 + (1-Y_pred_0) + (1-Y_pred_1) + (1-Y_pred_2) + (1-Y_pred_4);
% Y_pred(:,3) = Y_pred_4 + (1-Y_pred_0) + (1-Y_pred_1) + (1-Y_pred_2) + (1-Y_pred_3);
% 
% Y_pred = Y_pred ./ sum(Y_pred, 2);
% [~, Y_pred_labels] = max(Y_pred, [], 2);
% Y_pred_labels = Y_pred_labels - 1;
% 
% class_report_final = classperf(Y_test, Y_pred_labels);
% 
% % Calculate F1 score
% f1Score = 2 * (class_report_final.Sensitivity * class_report_final.PositivePredictiveValue) / (class_report_final.Sensitivity + class_report_final.PositivePredictiveValue);
% 
% % Calculate recall score
% recall = class_report_final.Sensitivity;
% 
% % Calculate 
% cm = confusionchart(Y_test,Y_pred_labels);
% 
% % Store score metrics
% accuracy_list_final = [accuracy_list_final, class_report_init.CorrectRate];
% f1score_list_final = [f1score_list_final, f1Score];
% recall_list_final = [recall_list_final, recall];
% cm_list_final = [cm_list_final, cm];

fprintf('\nInitial recall: %.1f%%', mean(recall_list_init)*100), '\n';
fprintf('\nInitial f1 score: %.1f%%', mean(f1score_list_init)*100), '\n';
fprintf('\nInitial matthews corr coefficient: %.3f%', mean(f1score_list_init)), '\n';
fprintf('\nInitial Cohens Kappa: %.3f%', mean(kappa_list_init)), '\n';
fprintf('\n')

toc