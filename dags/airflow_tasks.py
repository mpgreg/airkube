
from airflow.decorators import task

@task.virtualenv(python_version=3.8)
def setup_task(state_dict:dict)-> dict: 
    
    import snowflake.snowpark.functions as F
    from dags.snowpark_connection import snowpark_connect
    from dags.elt import reset_database

    session, _ = snowpark_connect('./include/state.json')
    reset_database(session=session, state_dict=state_dict, prestaged=True)

    _ = session.sql('CREATE STAGE '+state_dict['model_stage_name']).collect()
    _ = session.sql('CREATE TAG model_id_tag').collect()
    
     _ = session.use_warehouse(state_dict['compute_parameters']['load_warehouse'])

    print('Running initial bulk ingest from '+state_dict['connection_parameters']['download_base_url'])
    
    _ = bulk_elt(session=session, 
                 state_dict=state_dict, 
                 download_base_url=state_dict['connection_parameters']['download_base_url'],
                 use_prestaged=True)

    _ = materialize_holiday_table(session=session, 
                                  holiday_table_name=state_dict['holiday_table_name'])

    _ = subscribe_to_weather_data(session=session, 
                                  weather_database_name=state_dict['weather_database_name'], 
                                  weather_listing_id=state_dict['weather_listing_id'])

    _ = create_weather_view(session=session,
                            weather_table_name=state_dict['weather_table_name'],
                            weather_view_name=state_dict['weather_view_name'])

    _ = session.sql('CREATE STAGE IF NOT EXISTS ' + state_dict['model_stage_name']).collect()

    _ = deploy_eval_udf(session=session, 
                        udf_name=state_dict['eval_udf_name'],
                        function_name=state_dict['eval_func_name'],
                        model_stage_name=state_dict['model_stage_name'])
    session.close()

    return state_dict

@task.virtualenv(python_version=3.8)
def incremental_elt_task(state_dict: dict, files_to_download:list)-> dict:
    from dags.ingest import incremental_elt
    from dags.snowpark_connection import snowpark_connect

    session, _ = snowpark_connect()

    print('Ingesting '+str(files_to_download))
    download_base_url=state_dict['connection_parameters']['download_base_url']

    _ = session.use_warehouse(state_dict['compute_parameters']['load_warehouse'])

    _ = incremental_elt(session=session, 
                        state_dict=state_dict, 
                        files_to_ingest=files_to_download,
                        download_base_url=download_base_url,
                        use_prestaged=True)

    #_ = session.sql('ALTER WAREHOUSE IF EXISTS '+state_dict['compute_parameters']['load_warehouse']+\
    #                ' SUSPEND').collect()

    session.close()
    return state_dict

@task.virtualenv(python_version=3.8)
def generate_feature_table_task(state_dict:dict)-> dict:
    from dags.snowpark_connection import snowpark_connect
    from dags.mlops_pipeline import create_feature_table

    print('Generating features for all stations.')
    session, _ = snowpark_connect()

    session.use_warehouse(state_dict['compute_parameters']['fe_warehouse'])

    _ = session.sql("CREATE OR REPLACE TABLE "+state_dict['clone_table_name']+\
                    " CLONE "+state_dict['trips_table_name']).collect()
    _ = session.sql("ALTER TABLE "+state_dict['clone_table_name']+\
                    " SET TAG model_id_tag = '"+state_dict['model_id']+"'").collect()

    _ = create_feature_table(session, 
                             trips_table_name=state_dict['clone_table_name'], 
                             holiday_table_name=state_dict['holiday_table_name'], 
                             weather_view_name=state_dict['weather_view_name'],
                             feature_table_name=state_dict['feature_table_name'])

    _ = session.sql("ALTER TABLE "+state_dict['feature_table_name']+\
                    " SET TAG model_id_tag = '"+state_dict['model_id']+"'").collect()

    session.close()
    return state_dict

@task.virtualenv(python_version=3.8)
def generate_forecast_table_task(state_dict:dict)-> dict: 
    from dags.snowpark_connection import snowpark_connect
    from dags.mlops_pipeline import create_forecast_table

    print('Generating forecast features.')
    session, _ = snowpark_connect()

    _ = create_forecast_table(session, 
                              trips_table_name=state_dict['trips_table_name'],
                              holiday_table_name=state_dict['holiday_table_name'], 
                              weather_view_name=state_dict['weather_view_name'], 
                              forecast_table_name=state_dict['forecast_table_name'],
                              steps=state_dict['forecast_steps'])

    _ = session.sql("ALTER TABLE "+state_dict['forecast_table_name']+\
                    " SET TAG model_id_tag = '"+state_dict['model_id']+"'").collect()

    session.close()
    return state_dict

@task.virtualenv(python_version=3.8)
def batch_train_predict_task(feature_state_dict:dict, 
                             forecast_state_dict:dict)-> dict: 

    from dags.airflow_kubeflow import batch_train_pred_k8s
    
    print('Running Kubeflow Pytorch Job for batch training and inference')
    
    state_dict = batch_train_pred_k8s(feature_state_dict)
    
    return state_dict

@task.virtualenv(python_version=3.8)
def eval_station_models_task(state_dict:dict, 
                             pred_state_dict:dict,
                             run_date:str)-> dict:

    from dags.snowpark_connection import snowpark_connect
    from dags.mlops_pipeline import evaluate_station_model

    print('Running eval UDF for model output')
    session, _ = snowpark_connect()

    eval_table_name = evaluate_station_model(session, 
                                             run_date=run_date, 
                                             eval_model_udf_name=state_dict['eval_udf_name'], 
                                             pred_table_name=state_dict['pred_table_name'], 
                                             eval_table_name=state_dict['eval_table_name'])

    _ = session.sql("ALTER TABLE "+state_dict['eval_table_name']+\
                    " SET TAG model_id_tag = '"+state_dict['model_id']+"'").collect()
    session.close()
    return state_dict                                               
