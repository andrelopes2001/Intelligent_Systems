%% Load data

processed_data = readtable('dataframes\processed_df.csv');

X = table2array(processed_data(1:end, 4:end-2));
Y = processed_data.Type; 

% Define a mapping from category strings to numerical values
category_map = containers.Map({'F', 'N', 'Q', 'S', 'V'}, {0, 1, 2, 3, 4});

% Use the mapping to convert the categorical variable to numerical values
Y = cellfun(@(x) category_map(x), Y);

partition = cvpartition(Y, 'KFold', 5, 'Stratify', true);

accuracy_list = [];
f1score_list = [];

for i = 1:5
    train_idx = training(partition, i); % Get the training indices for this fold
    test_idx = test(partition, i); % Get the test indices for this fold

    % Extract the training and test data for this fold
    X_train = X(train_idx, :);
    Y_train = Y(train_idx, :);
    X_test = X(test_idx, :);
    Y_test = Y(test_idx, :);
    
    opt = genfisOptions('FCMClustering', 'FISType', 'sugeno');
    opt.NumClusters = 5;

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
    Y_pred(:,2) = Y_pred_3 + (1-Y_pred_0) + (1-Y_pred_1) + (1-Y_pred_2) + (1-Y_pred_4);
    Y_pred(:,3) = Y_pred_4 + (1-Y_pred_0) + (1-Y_pred_1) + (1-Y_pred_2) + (1-Y_pred_3);

    Y_pred = Y_pred ./ sum(Y_pred, 2);
    [~, Y_pred_labels] = max(Y_pred, [], 2);
    Y_pred_labels = Y_pred_labels - 1;

    class_report = classperf(Y_test, Y_pred_labels);

    % Calculate F1 score
    f1Score = 2 * (class_report.Sensitivity * class_report.PositivePredictiveValue) / (class_report.Sensitivity + class_report.PositivePredictiveValue);

    % Store accuracies and f1 scores
    accuracy_list = [accuracy_list, class_report.CorrectRate];
    f1score_list = [f1score_list, f1Score];

end

fprintf('Accuracy: %.1f%%\n', mean(accuracy_list)*100);
fprintf('F1 Score: %.1f%%\n', mean(f1score_list)*100);
