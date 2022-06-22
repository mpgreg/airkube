
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from dags.airflow_tasks import setup_task
from dags.airflow_tasks import generate_feature_table_task
from dags.airflow_tasks import generate_forecast_table_task
from dags.airflow_tasks import batch_train_predict_task
#from dags.airflow_tasks import eval_station_models_task 


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

#local_airflow_path = '/usr/local/airflow/'

@dag(default_args=default_args, schedule_interval=None, start_date=datetime(2020, 3, 1), catchup=False, tags=['setup'])
def citibikeml_setup_taskflow(run_date:str):
    """
    Setup initial Snowpark / Astronomer ML Demo
    """
    import uuid
    import json
    
    with open('./include/state.json') as sdf:
        state_dict = json.load(sdf)
    
    model_id = str(uuid.uuid1()).replace('-', '_')

    state_dict.update({'model_id': model_id})
    state_dict.update({'run_date': run_date})
    state_dict.update({'weather_database_name': 'WEATHER_NYC'})
    state_dict.update({'load_table_name': 'RAW_',
                       'trips_table_name': 'TRIPS',
                       'load_stage_name': 'LOAD_STAGE',
                       'model_stage_name': 'MODEL_STAGE',
                       'weather_table_name': state_dict['weather_database_name']+'.ONPOINT_ID.HISTORY_DAY',
                       'weather_view_name': 'WEATHER_NYC_VW',
                       'holiday_table_name': 'HOLIDAYS',
                       'clone_table_name': 'CLONE_'+model_id,
                       'feature_table_name' : 'FEATURE_'+model_id,
                       'pred_table_name': 'PRED_'+model_id,
                       'eval_table_name': 'EVAL_'+model_id,
                       'forecast_table_name': 'FORECAST_'+model_id,
                       'forecast_steps': 30,
                       'train_udf_name': 'station_train_predict_udf',
                       'train_func_name': 'station_train_predict_func',
                       'eval_udf_name': 'eval_model_output_udf',
                       'eval_func_name': 'eval_model_func',
                       'model_file_name' : 'forecast_model.zip',
                       'le_file_name' : 'label_encoders.pkl',
                       'cat_cols' : ['STATION_ID', 'HOLIDAY'],
                       'k8s_namespace' : 'citibike',
                       'train_image' : 'docker.io/mpgregor/airkube:latest',
                       'train_job_name' : 'citibike-train-'+model_id.replace('_', '-').lower()
                      })
    state_dict['connection_parameters']['download_base_url'] = 's3://sfquickstarts/vhol_citibike_ml_snowpark_python/data'
    
    #Task order - one-time setup
    setup_state_dict = setup_task(state_dict)
    feature_state_dict = generate_feature_table_task(setup_state_dict) 
    forecast_state_dict = generate_forecast_table_task(setup_state_dict)
    pred_state_dict = batch_train_predict_task(feature_state_dict, forecast_state_dict)
    
    #eval_state_dict = eval_station_models_task(eval_udf_state_dict, pred_state_dict, run_date)  

    return state_dict

run_date='2020_01_01'

state_dict = citibikeml_setup_taskflow(run_date=run_date)
