import argparse

def load_and_encode(state_dict):

    from snowflake import snowpark as snp
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from collections import defaultdict
    import pickle
    
    session = snp.Session.builder.configs(state_dict['connection_parameters']).create()
    session.use_warehouse(state_dict['compute_parameters']['default_warehouse'])

    feature_df = session.table(state_dict['feature_table_name']).to_pandas()
    #forecast_df = session.table(state_dict['forecast_table_name']).to_pandas()

    session.close()

    feature_df['DATE'] = pd.to_datetime(feature_df['DATE'])
    feature_df.set_index('DATE', inplace=True)
    
    #forecast_df['DATE'] = pd.to_datetime(forecast_df['DATE'])
    #forecast_df.set_index('DATE', inplace=True)

    cat_cols = state_dict['cat_cols']
    num_cols = [set(feature_df.columns)-set(cat_cols)]
    state_dict['num_cols'] = num_cols

    try:
        with open(state_dict['le_file_name'], 'rb') as fh: 
            d=pickle.load(fh)
        feature_df[cat_cols]=feature_df[cat_cols].apply(lambda x: d[x.name].transform(x))

    except: 
        d = defaultdict(LabelEncoder)
        feature_df[cat_cols]=feature_df[cat_cols].apply(lambda x: d[x.name].fit_transform(x))

        with open(state_dict['le_file_name'], 'wb') as fh: 
            pickle.dump(d, fh)

    return state_dict, feature_df

def train_and_save(state_dict, feature_df):
    import pandas as pd
    from pytorch_tabnet.tab_model import TabNetRegressor
    
    feature_df.sort_values(by='DATE', ascending=True, inplace=True)

    train_df = feature_df.groupby('STATION_ID').head(-365)
    valid_df = feature_df.groupby('STATION_ID').tail(365)

    state_dict['cat_idxs'] = [feature_df.drop(columns=['COUNT'], axis=1).columns.get_loc(col) for col in state_dict['cat_cols']]
    state_dict['cat_dims'] = list(feature_df.drop(columns=['COUNT'], axis=1).iloc[:, state_dict['cat_idxs']].nunique().values)

    y_train = train_df['COUNT'].values.reshape(-1,1)
    X_train = train_df.drop(columns ='COUNT', axis=1).values

    y_valid = valid_df['COUNT'].values.reshape(-1,1)
    X_valid = valid_df.drop(columns ='COUNT', axis=1).values
    
    model = TabNetRegressor(cat_idxs=state_dict['cat_idxs'], cat_dims=state_dict['cat_dims'])

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        max_epochs=1,
        patience=100,
        batch_size=2048, 
        virtual_batch_size=256,
        num_workers=0,
        drop_last=True)

    model.save_model(state_dict['model_file_name'].split('.')[0])
    
    return state_dict

def pred(state_dict, feature_df):
    from pytorch_tabnet.tab_model import TabNetRegressor
    import pandas as pd
    from torch import tensor
    
    model = TabNetRegressor(cat_idxs=state_dict['cat_idxs'], cat_dims=state_dict['cat_dims'])

    model.load_model(state_dict['model_file_name'])
    
    pred_df = feature_df.copy(deep=True)
    
    pred_df['PRED'] = model.predict(tensor(feature_df.drop(columns=['COUNT']).values)).round().astype('int')
    
    return state_dict, pred_df

def forecast(state_dict, feature_df, forecast_df):

    if len(state_dict['lag_values']) > 0:
        for step in range(state_dict['forecast_steps']):
            #station_id = df.iloc[-1]['STATION_ID']
            future_date = df.iloc[-1]['DATE']+timedelta(days=1)
            lags=[df.shift(lag-1).iloc[-1]['COUNT'] for lag in state_dict['lag_values']]
            forecast=forecast_df.loc[forecast_df['DATE']==future_date.strftime('%Y-%m-%d')]
            forecast=forecast.drop(labels='DATE', axis=1).values.tolist()[0]
            features=[*lags, *forecast]
            pred=round(model.predict(np.array([features]))[0][0])
            row=[future_date, pred, *features, pred]
            df.loc[len(df)]=row

    return state_dict, pred_df

def decode_and_write(state_dict, pred_df):
    from snowflake import snowpark as snp
    import pandas as pd
    import pickle
    
    with open(state_dict['le_file_name'], 'rb') as fh: 
        d=pickle.load(fh)

    pred_df[state_dict['cat_cols']] = pred_df[state_dict['cat_cols']].apply(lambda x: d[x.name].inverse_transform(x))

    session = snp.Session.builder.configs(state_dict['connection_parameters']).create()
    session.use_warehouse(state_dict['compute_parameters']['default_warehouse'])

    session.create_dataframe(pred_df).write.mode('overwrite').save_as_table(state_dict['pred_table_name'])
    
    session.close()
    
    return state_dict

if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='airkube training')
    parser.add_argument('--password', type=str)
    parser.add_argument('--account', type=str)
    parser.add_argument('--username', type=str)
    parser.add_argument('--role', type=str)
    parser.add_argument('--database', type=str)
    parser.add_argument('--schema', type=str)
    parser.add_argument('--feature_table_name', type=str)
    parser.add_argument('--pred_table_name', type=str)
    
    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    #Path(args.accuracy).parent.mkdir(parents=True, exist_ok=True)

    state_dict = {"connection_parameters": {"password": args.password},
                  "compute_parameters" : {"default_warehouse": "XSMALL_WH"}}
    state_dict['connection_parameters']['user'] = args.username
    state_dict['connection_parameters']['account'] = args.account
    state_dict['connection_parameters']['role'] = args.role
    state_dict['connection_parameters']['database'] = args.database
    state_dict['connection_parameters']['schema'] = args.schema
    state_dict['feature_table_name'] = args.feature_table_name
    state_dict['pred_table_name'] = args.pred_table_name
    state_dict['model_file_name']='forecast_model.zip'
    state_dict['le_file_name']='label_encoders.pkl'
    state_dict["cat_cols"] = ['STATION_ID', 'HOLIDAY']

    load_state_dict, feature_df = load_and_encode(state_dict)
    train_state_dict = train_and_save(load_state_dict, feature_df)
    pred_state_dict, pred_df = pred(state_dict, feature_df)
    state_dict = decode_and_write(state_dict, pred_df)

