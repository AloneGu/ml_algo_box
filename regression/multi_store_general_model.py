import os
import csv
from backend_common.storage import storage_manager
import numpy as np
from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.cross_validation import train_test_split

STORAGE_TYPE = 2
BUCKET_NAME = 'percolata-data'
REMOTE_TRAINING_DIR = 'data/combined/fusion_training'
MODEL = 'linear_regression'


def get_training_data(location_id):
    remote_dir = storage_manager.generate_storage_path(STORAGE_TYPE, BUCKET_NAME, REMOTE_TRAINING_DIR)
    fl = storage_manager.get_list_of_remote_path(remote_dir)
    # TODO: choose training data based on time
    for f in fl:
        fn = os.path.basename(f)
        if fn.startswith(str(location_id)):
            tmp_local_dir = '/tmp/fusion_training/'
            if not os.path.exists(tmp_local_dir):
                os.makedirs(tmp_local_dir)
            tmp_path = os.path.join(tmp_local_dir, fn)
            storage_manager.download_to_local_path(f, tmp_path)
            local_f_handler = open(tmp_path, 'r')
            csv_r = csv.reader(local_f_handler)
            training_data = []
            first_line_flag = True
            for r in csv_r:
                if first_line_flag:
                    first_line_flag = False
                    continue
                try:
                    training_data.append(map(int,r[-6:]))
                except:
                    pass
            local_f_handler.close()
            os.remove(tmp_path)
            return training_data

def calc_accuracy(a,b):
    err_rate = []
    for x,y in zip(a,b):
        y = max(round(y),0) # change y to non-neg int
        if x == 0:
            tmp_err_rate = 0 if y==0 else 1
        else:
            tmp_err_rate = abs(x-y)/x
        err_rate.append(tmp_err_rate)
    return round(np.average(err_rate),2)

def calc_mse(a,b):
    err_rate = []
    for x,y in zip(a,b):
        y = max(round(y),0) # change y to non-neg int
        err_rate.append(abs(x-y)**2)
    return round(np.average(err_rate),2)


class TrafficFusionProcessor(object):
    def __init__(self, location_ids):
        self.location_ids = location_ids.split(',')

    def run(self):

        for location_id in self.location_ids:
            training_data = get_training_data(location_id)
        x,y=[],[]
        for r in training_data:
            x.append([r[2],r[4]])
            y.append(r[0])


        model_dict = {'linear':LinearRegression(),
                      'bayesianridge':BayesianRidge(),
                      'ridge':Ridge(),
                      'lasso':Lasso(),
                      'randomforest':RandomForestRegressor(),
                      'adaboost':AdaBoostRegressor(),
                      'gradientboost':GradientBoostingRegressor(),
                      'extratree':ExtraTreesRegressor()
                      }

        # prepare train data and test data
        data_dict = {}
        for i in range(1,18):
            curr_rate = 0.05*(1+i)
            x_t,x_test,y_t,y_test = train_test_split(x,y,train_size=curr_rate)
            data_dict[i] = [x_t,x_test,y_t,y_test]



        res_data = []
        for model_name in model_dict:
            train_rate_list,err_rate_list,mse,test_err_list,test_mse = [],[],[],[],[]
            for i in range(1,18):
                curr_rate = 0.05*(1+i)
                train_rate_list.append(round(curr_rate,2))
                x_t,x_test,y_t,y_test = data_dict[i]
                # fit model
                m = model_dict[model_name]
                m.fit(x_t,y_t)
                # on whole data set
                res = m.predict(x)
                curr_err_rate = calc_accuracy(y,res)
                err_rate_list.append(100*curr_err_rate)
                mse.append(calc_mse(y,res))
                # on test data set
                test_res = m.predict(x_test)
                test_err_rate = calc_accuracy(y_test,test_res)
                test_err_list.append(100*test_err_rate)
                test_mse.append(calc_mse(y_test,test_res))
            res_data.append([train_rate_list,err_rate_list,mse,model_name,test_err_list,test_mse])

        import matplotlib.pyplot as plt
        # graph whole data set mse
        plt.figure(figsize=(20,10))
        plt.xlim([0,1])
        for train_rate_list,err_rate_list,mse,model_name,_,_ in res_data:
            plt.plot(train_rate_list,mse,label=model_name)
            plt.title('location_id:'+str(self.location_id)+'  all walkin data rows:'+str(len(training_data)))
            plt.ylabel('whole data set mse')
            plt.xlabel('train set rate')
            plt.legend()
            plt.savefig('./analyse_graph/'+str(self.location_id)+'_whole_mse.jpg')
        # graph test data set mse
        plt.figure(figsize=(20,10))
        plt.xlim([0,1])
        for train_rate_list,_,_,model_name,err_rate_list,mse in res_data:
            plt.plot(train_rate_list,mse,label=model_name)
            plt.title('location_id:'+str(self.location_id)+'  all walkin data rows:'+str(len(training_data)))
            plt.ylabel('test data set mse')
            plt.xlabel('train set rate')
            plt.legend()
            plt.savefig('./analyse_graph/'+str(self.location_id)+'_test_mse.jpg')
        # graph whole data set err percent
        plt.figure(figsize=(20,10))
        plt.xlim([0,1])
        for train_rate_list,err_rate_list,mse,model_name,_,_ in res_data:
            plt.plot(train_rate_list,err_rate_list,label=model_name)
            plt.title('location_id:'+str(self.location_id)+'  all walkin data rows:'+str(len(training_data)))
            plt.ylabel('whole data set err percentage')
            plt.xlabel('train set rate')
            plt.legend()
            plt.savefig('./analyse_graph/'+str(self.location_id)+'_whole_err_rate.jpg')
        # graph test data set err percent
        plt.figure(figsize=(20,10))
        plt.xlim([0,1])
        for train_rate_list,_,_,model_name,err_rate_list,mse in res_data:
            plt.plot(train_rate_list,err_rate_list,label=model_name)
            plt.title('location_id:'+str(self.location_id)+'  all walkin data rows:'+str(len(training_data)))
            plt.ylabel('test data set err percentage')
            plt.xlabel('train set rate')
            plt.legend()
            plt.savefig('./analyse_graph/'+str(self.location_id)+'_test_err_rate.jpg')



def traffic_fusion_process(locations):
    # entry
    traffic_processor = TrafficFusionProcessor(l)
    traffic_processor.run()



if __name__ == '__main__':
   traffic_fusion_process('99,104,108,109,111,113,116,120,123,133')

