from collections import defaultdict
from tokenizer import tokenizer
#here we add the data to the inverse index by tokenizing the words and then adding it to the dictionary
def inverse_index(data):
    d_map = defaultdict(list)

    for idx, val in enumerate(data):
        for word in tokenizer(val):
            d_map[word].append(idx)


            
            import os
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import time
from pathlib import Path
import sys
sys.path.append(YOUR DIRECTORY)
import confusion_matrix_plot
     
# *******************************************************************
         
def create_save_models_bayes_opt(dtrain, dvalid, error_metric, 
                                 num_random_points, num_iterations, 
                                 results_dir, processor_type,
                                 number_of_classes):
    # xgb single params
    dict_single_parameters = {'booster':'gbtree', 'verbosity':0,
                              'objective':'multi:softprob',
                              'eval_metric':error_metric,  # 'mlogloss',
                              'num_class':number_of_classes}
     
    if processor_type == 'cpu':
        tree_method  = 'hist' # auto
        predictor = 'auto'  # 'cpu_predictor' , default = auto
    elif processor_type == 'gpu':
        tree_method  = 'gpu_hist'
        predictor = 'gpu_predictor' # default = auto
    else:
        print('\ninvalid processor_type in create_models():',processor_type)
        raise NameError
         
    dict_single_parameters['tree_method'] = tree_method
    dict_single_parameters['predictor'] = predictor
     
    # save to results dir
    dict_file = open(results_dir + 'dict_single_parameters.pkl','wb')
    pickle.dump(dict_single_parameters, dict_file)
        
    maximum_boosting_rounds = 1500
    early_stop_rounds = 10
    list_boosting_rounds = []
     
    start_time_total = time.time()
     
    def xgb_function(eta, max_depth, colsample_bytree, reg_lambda, reg_alpha,
                     subsample, max_bin):
         
        dict_parameters = {'eta':eta, 'max_depth':int(max_depth), 
        'colsample_bytree':colsample_bytree, 'reg_lambda':int(reg_lambda), 
        'reg_alpha':int(reg_alpha), 'subsample':subsample, 'max_bin':int(max_bin)}
         
        dict_parameters.update(dict_single_parameters)        
         
        watchlist = [(dvalid, 'eval')]
        xgb_model = xgb.train(params=dict_parameters, dtrain=dtrain, evals=watchlist,
                              num_boost_round=maximum_boosting_rounds,
                              early_stopping_rounds=early_stop_rounds,
                              verbose_eval=False)
         
        multiclass_log_loss = xgb_model.best_score
        boosting_rounds = xgb_model.best_ntree_limit
         
        # record boosting_rounds here, added to bayes opt parameter df below
        list_boosting_rounds.append(boosting_rounds)
         
        # bayes opt is a maximization algorithm, to minimize mlogloss, return 1-this
        bayes_opt_score = 1.0 - multiclass_log_loss
         
        return bayes_opt_score
         
    optimizer = BayesianOptimization(f=xgb_function,
                pbounds={'eta': (0.1, 0.8),
                         'max_depth': (3, 21),
                         'colsample_bytree': (0.5, 0.99),
                         'reg_lambda': (1, 6),
                         'reg_alpha': (0, 3),
                         'subsample': (0.6, 0.9),
                         'max_bin': (32, 257)},
                         verbose=2)
     
    optimizer.maximize(init_points=num_random_points, 
                       n_iter=num_iterations)
     
    print('\nbest result:', optimizer.max)
     
    elapsed_time_total = (time.time()-start_time_total)/60
    print('\n\ntotal elapsed time =',elapsed_time_total,' minutes')
     
    # optimizer.res is a list of dicts
    list_dfs = []
    counter = 0
    for result in optimizer.res:
        df_temp = pd.DataFrame.from_dict(data=result['params'], orient='index',
                                         columns=['trial' + str(counter)]).T
        df_temp['bayes opt error'] = result['target']
         
        df_temp['boosting rounds'] = list_boosting_rounds[counter]
         
        list_dfs.append(df_temp)
         
        counter = counter + 1
         
    df_results = pd.concat(list_dfs, axis=0)
    df_results.to_pickle(results_dir + 'df_bayes_opt_results_parameters.pkl')
    df_results.to_csv(results_dir + 'df_bayes_opt_results_parameters.csv')
         
# end of create_save_models_bayes_opt()
     
# *******************************************************************
             
def make_final_predictions(dcalib, dprod, yprod, 
                           list_class_names, models_directory, 
                           save_directory, save_models_flag, df_params,
                           threshold, ml_name, dict_single_params):
     
    # apply threshold
    accepted_models_num = 0
    list_predicted_prob = []
    num_models = df_params.shape[0]
    for i in range(num_models):
        if df_params.loc[df_params.index[i],'bayes opt error'] > threshold:
             
            dict_temp = {'eta':df_params.loc[df_params.index[i],'eta'], 
            'max_depth':int(df_params.loc[df_params.index[i],'max_depth']), 
            'colsample_bytree':df_params.loc[df_params.index[i],'colsample_bytree'], 
            'reg_lambda':int(df_params.loc[df_params.index[i],'reg_lambda']), 
            'reg_alpha':int(df_params.loc[df_params.index[i],'reg_alpha']), 
            'subsample':df_params.loc[df_params.index[i],'subsample'], 
            'max_bin':int(df_params.loc[df_params.index[i],'max_bin'])}
             
            dict_temp.update(dict_single_params)       
             
             
            ml_model = xgb.train(params=dict_temp, dtrain=dcalib,
                       num_boost_round=df_params.loc[df_params.index[i],'boosting rounds'],
                       verbose_eval=False)
             
            # these are class probabilities
            list_predicted_prob.append(ml_model.predict(dprod))
             
            accepted_models_num = accepted_models_num + 1
             
            if save_models_flag:
                model_name = ml_name + df_params.index[i] + '.joblib'
                joblib.dump(ml_model, save_directory + model_name)
 
    # compute mean probabilities
    mean_probabilities = np.mean(list_predicted_prob, axis=0)
     
    # compute predicted class
    # argmax uses 1st ocurrance in case of a tie
    y_predicted_class = np.argmax(mean_probabilities, axis=1)
     
    # compute and save error measures
 
    # print info to file
    stdout_default = sys.stdout
    sys.stdout = open(save_directory + ml_name + '_prediction_results.txt','w')
     
    print('balanced accuracy score =',balanced_accuracy_score(yprod, y_predicted_class))
     
    print('accuracy score =',accuracy_score(yprod, y_predicted_class))
     
    print('number of accepted models =',accepted_models_num,' for threshold =',threshold)
     
    print('\nclassification report:')
    print(classification_report(yprod, y_predicted_class, digits=3, output_dict=False))
     
    print('\nraw confusion matrix:')
    cm_raw = confusion_matrix(yprod, y_predicted_class)
    print(cm_raw)
     
    print('\nconfusion matrix normalized by prediction:')
    cm_pred = confusion_matrix(yprod, y_predicted_class, normalize='pred')
    print(cm_pred)
     
    print('\nconfusion matrix normalized by true:')
    cm_true = confusion_matrix(yprod, y_predicted_class, normalize='true')
    print(cm_true)
     
    sys.stdout = stdout_default
     
    # plot and save confustion matrices
    figure_size = (12, 8)
    number_of_decimals = 4
     
    confusion_matrix_plot.confusion_matrix_save_and_plot(cm_raw, 
    list_class_names, save_directory, 'Confusion Matrix', 
    ml_name + '_confusion_matrix', False, None, 30, figure_size,
    number_of_decimals)
     
    confusion_matrix_plot.confusion_matrix_save_and_plot(cm_pred, 
    list_class_names, save_directory, 'Confusion Matrix Normalized by Prediction', 
    ml_name + '_confusion_matrix_norm_by_prediction', False, 'pred', 
    30, figure_size, number_of_decimals)
     
    confusion_matrix_plot.confusion_matrix_save_and_plot(cm_true, 
    list_class_names, save_directory, 'Confusion Matrix Normalized by Actual', 
    ml_name + '_confusion_matrix_norm_by_true', False, 'true', 
    30, figure_size, number_of_decimals)
 
# end of make_final_predictions()
        
# *******************************************************************
 
if __name__ == '__main__':    
     
    type_of_processor = 'gpu'
     
    ml_algorithm_name = 'xgb'
    file_name_stub = ml_algorithm_name + '_bayes_opt' 
     
    calculation_type = 'production' #'calibration' 'production'
     
    data_directory = YOUR DIRECTORY
     
    base_directory = YOUR DIRECTORY
     
    results_directory_stub = base_directory + file_name_stub + '/'
    if not Path(results_directory_stub).is_dir():
        os.mkdir(results_directory_stub)
                
    # fixed parameters
    error_type = 'mlogloss'  # multi class log loss
    threshold_error = 0.9284  #this is 1 - mlogloss
    total_number_of_iterations = 40
    number_of_random_points = 10  # random searches to start opt process
    # this is # of bayes iters, thus total=this + # of random pts
    number_of_iterations =  total_number_of_iterations - number_of_random_points
    save_models = False
     
        
    # read data
    x_calib = np.load(data_directory + 'x_mnist_calibration.npy')        
    y_calib = np.load(data_directory + 'y_mnist_calibration.npy')
    d_calib = xgb.DMatrix(x_calib, label=y_calib)
    num_classes = np.unique(y_calib).shape[0]
     
    x_train, x_valid, y_train, y_valid = train_test_split(x_calib ,y_calib,
    train_size=0.75, shuffle=True, stratify=y_calib)
     
    # transform numpy arrays into dmatrix format
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
                 
    print('\n*** starting at',pd.Timestamp.now())
 
    if calculation_type == 'calibration':
         
        results_directory = results_directory_stub + calculation_type + '/'
        if not Path(results_directory).is_dir():
            os.mkdir(results_directory)
         
        create_save_models_bayes_opt(d_train, d_valid, error_type,
                                     number_of_random_points, number_of_iterations,
                                     results_directory, type_of_processor,
                                     num_classes)
                           
    elif calculation_type == 'production':
         
        # get xgboost parameters
        models_dir = results_directory_stub + 'calibration/'
        df_parameters = pd.read_pickle(models_dir + 'df_bayes_opt_results_parameters.pkl')
         
        dict_file = open(models_dir + 'dict_single_parameters.pkl','rb')
        dictionary_single_parameters = pickle.load(dict_file)
         
        results_directory = results_directory_stub + calculation_type + '/'
        if not Path(results_directory).is_dir():
            os.mkdir(results_directory)
             
        x_prod = np.load(data_directory + 'x_mnist_production.npy')
        y_prod = np.load(data_directory + 'y_mnist_production.npy')
         
        num_classes = np.unique(y_prod).shape[0]
        class_names_list = []
        for i in range(num_classes):
            class_names_list.append('class ' + str(i))
     
        # transform numpy arrays into dmatrix format
        d_prod = xgb.DMatrix(x_prod, label=y_prod)
        d_calib = xgb.DMatrix(x_calib, label=y_calib)
                 
        make_final_predictions(d_calib, d_prod, y_prod, class_names_list, 
                               models_dir, results_directory, 
                               save_models, df_parameters, 
                               threshold_error, ml_algorithm_name, 
                               dictionary_single_parameters)
               
    else:
        print('\ninvalid calculation type:',calculation_type)
        raise NameError
