import numpy as np
import pandas as pd
import shutil, os
from os import path
import sqlite3
import json
from sklearn.externals import joblib
from IPython.core.display import display
import copy as cp
from functools import partial
import networkx as nx
import inspect
from sklearn import grid_search
from collections import defaultdict

from models import *

## low IO api, helper function
def overwrite_to_db(db, table, row):
    """
    db: sqlite3 datafile path
    tablename: table in the database
    row: dict of {col:value}
    """
    conn = sqlite3.connect(db)
    c = conn.cursor()
    ## name is unique id in both data and model db
    #update_statement = "UPDATE %s SET %s WHERE name='%s'" % (table, ','.join(['%s=%s' % (k,v) for k,v in row.items()]), row['name'])
    delete_statement = "DELETE FROM %s WHERE name='%s'" % (table, row['name'])
    insert_statement = """INSERT INTO %s (%s) VALUES (%s)""" % (table, 
                                                         ','.join(row.keys()), 
                                                         ','.join(["'%s'" % (s,) for s in row.values()]))


    c.execute(delete_statement) ## not ideal but fast enough for prototyping
    c.execute(insert_statement)
    conn.commit()
    conn.close()
    
def query_db(db, table, columns = '*', where = None):
    """
    return a list of dict (with columns as KEYS, and query results as VALUES)
    """
    conn = sqlite3.connect(db)
    c = conn.cursor()
    where = ' where ' + where if where else ''
    statement = "SELECT %s from %s%s" % (','.join(columns), table, where)
    rows = c.execute(statement)
    column_names = columns if columns != '*' else [x[0] for x in c.description]
    results = [dict(zip(column_names, row)) for row in rows]
    conn.commit()
    conn.close()
    return results

def write_item(meta, bulk, item_type, project_path = None):
    """
    write data/model to the meta database and binary files
    """
    prefix = 'data_' if item_type == 'data' else 'model_'
    ## find the way
    item_folder = get_config(prefix+'folder', project_path)
    item_meta_db = get_config(prefix+'meta_db', project_path)
    item_meta_table = get_config(prefix+'meta_table', project_path)
    ## insert meta information to meta.db
    overwrite_to_db(db = item_meta_db, table = item_meta_table, row = meta)
    ## save data_bulk into database
    bulk_folder = path.join(item_folder, meta['name'])
    if path.exists(bulk_folder):
        shutil.rmtree(bulk_folder)
    os.mkdir(bulk_folder)
    bulk_file = path.join(bulk_folder, 'bulk.pkl')
    joblib.dump(bulk, bulk_file)
    
def read_meta_by_name(item_name, item_type, project_path = None):
    """
    return meta informatino of item from the corresponding meta database
    """
    prefix = 'data_' if item_type == 'data' else 'model_'
    ## find the way
    item_folder = get_config(prefix+'folder', project_path)
    item_meta_db = get_config(prefix+'meta_db', project_path)
    item_meta_table = get_config(prefix+'meta_table', project_path)
    ## query the database
    results = query_db(item_meta_db, item_meta_table, columns='*', where = 'name = "%s"' % item_name)
    return results[0] if results else None

def read_bulk_by_name(item_name, item_type, project_path = None):
    """
    load the model/data bulk into the memory and return it
    """
    prefix = 'data_' if item_type == 'data' else 'model_'
    ## find the way to bulk
    type_folder = get_config(prefix+'folder', project_path)
    item_file = path.join(type_folder, item_name, 'bulk.pkl')
    item = joblib.load(item_file)
    return item

def hglue(*dfs):
    """horizontally stack all the df in dfs
    example: xy_df = hglue([X, y]) 
    """
    return pd.concat(dfs, axis = 1)

def get_config(item, project_path = None):
    project_path = project_path or '.' ## stupid tradeoff, should be improved LATER!!
    CONFIG = {  'data_folder': path.abspath(path.join(project_path, 'data'))
              , 'model_folder': path.abspath(path.join(project_path, 'models'))
              , 'temp_folder': path.abspath(path.join(project_path, 'temp'))
              , 'data_meta_db': path.abspath(path.join(project_path, 'data/meta.db'))
              , 'data_meta_table': 'data_meta'
              , 'data_meta_schema': """(name text PRIMARY KEY, namespace text, input_features text, 
                                      output_features text, type integer)"""
              , 'model_meta_db': path.abspath(path.join(project_path, 'models/meta.db'))
              , 'model_meta_table': 'model_meta'
              , 'model_meta_schema': '(name text PRIMARY KEY, type integer, train_data type)'
              , 'data_signature_template': ('namespace', 'input_features', 'output_features', 'type')}
    return CONFIG[item]

class DataType(object):
    """data type 
    PRINCIPLE: THEY MUST BE EXCLUSIVE OF EACH
    """
    UNSUPERVISED = 2 ** 0
    BINARY_CLASSIFICATION = 2 ** 1
    MULTI_CLASSIFICATION = 2 ** 2
    REGRESSION = 2 ** 3
class ModelType(object):
    """model type
    which type of model can handle which set of data types 
    """
    BINARY_CLASSIFIER = DataType.BINARY_CLASSIFICATION
    MULTI_CLASSIFIER = DataType.BINARY_CLASSIFICATION + DataType.MULTI_CLASSIFICATION ## or relationship
    REGRESSOR = DataType.REGRESSION
    ## e.g. clustering, auto_encoder, or feature_selector or dim-reductor
    FEATURE_EXTRACTOR = (DataType.UNSUPERVISED + DataType.BINARY_CLASSIFICATION 
                         + DataType.MULTI_CLASSIFICATION + DataType.REGRESSION)
    ## e.g. missing value imputator
    PREPROCESSOR = (DataType.UNSUPERVISED + DataType.BINARY_CLASSIFICATION 
                         + DataType.MULTI_CLASSIFICATION + DataType.REGRESSION)
    # subsampling rows
    SUBSAMPLER = (DataType.UNSUPERVISED + DataType.BINARY_CLASSIFICATION 
                         + DataType.MULTI_CLASSIFICATION + DataType.REGRESSION)

## models and data

def write_project(container_path, project_name):
    ## check if project exists - overwrite
    project_path = path.abspath(path.join(container_path, project_name))
    if path.exists(project_path):
        shutil.rmtree(project_path)
    ## create folder for project
    os.mkdir(project_path)
    data_folder = get_config('data_folder', project_path)
    model_folder = get_config('model_folder', project_path)
    temp_folder = get_config('temp_folder', project_path)
    ## data folder
    os.mkdir(data_folder)
    data_meta_db = get_config('data_meta_db', project_path)
    data_meta_table = get_config('data_meta_table', project_path)
    data_meta_schema = get_config('data_meta_schema', project_path)
    conn = sqlite3.connect(data_meta_db)
    c = conn.cursor()
    c.execute("CREATE TABLE %s %s" % (data_meta_table, data_meta_schema))
    conn.commit()
    conn.close()   
    ## models folder
    os.mkdir(model_folder)
    model_meta_db = get_config('model_meta_db', project_path)
    model_meta_table = get_config('model_meta_table', project_path)
    model_meta_schema = get_config('model_meta_schema', project_path)
    conn = sqlite3.connect(model_meta_db)
    c = conn.cursor()
    c.execute("CREATE TABLE %s %s" % (model_meta_table, model_meta_schema))
    conn.commit()
    conn.close()
    ## temp folder
    os.mkdir(temp_folder)
    return project_path

def create_model_meta(model_name, model_type):
    return {'name': model_name, 
            'type': model_type}

def create_data_meta(data_name, data_namespace, input_feats, output_feats, data_type):
    return {  'name': data_name
            , 'namespace': data_namespace
            , 'input_features': input_feats
            , 'output_features': output_feats
            , 'type': data_type}
    
def write_data(data_meta, data_bulk, project_path = None):
    """
    """
    meta = cp.deepcopy(data_meta)
    ## JSON can only serialize built in list
    meta['input_features'] = json.dumps([fi for fi in meta['input_features']])
    meta['output_features'] = json.dumps([fo for fo in meta['output_features']])
    write_item(project_path = project_path, meta = meta, bulk = data_bulk, item_type = 'data')
    
def write_model(model_meta, model_bulk, project_path = None):
    """
    """
    write_item(project_path = project_path, meta = model_meta, bulk = model_bulk, item_type = 'model')
    
def read_data_meta(data_name, project_path = None):
    """
    return dictionary of data meta, as in the data/meta.db/data_meta table
    """
    meta = read_meta_by_name(project_path = project_path, item_name = data_name, item_type = 'data')
    meta['input_features'] = json.loads(meta['input_features'])
    meta['output_features'] = json.loads(meta['output_features'])
    return meta

def read_model_meta(model_name, project_path = None):
    """
    """
    return read_meta_by_name(project_path = project_path, item_name = model_name, item_type = 'model')

def read_data_bulk(data_name, project_path = None):
    """
    return the bulk of data as what it is saved as
    """
    return read_bulk_by_name(project_path = project_path, item_name = data_name, item_type = 'data')

def read_model_bulk(model_name, project_path = None):
    """
    return the bulk of data as what it is saved as
    """
    return read_bulk_by_name(project_path = project_path, item_name = model_name, item_type = 'model')

def trainable(model_meta, data_meta):
    """
    to test if model trainable on certain data
    RULE: model_type mactchs data_type
    """
    ## compare type information
    model_type, data_type = model_meta['type'], data_meta['type']
    match = (model_type & data_type > 0)
    return match

def predictable(model_meta, data_meta, project_path = None):
    """to test if model can be used to predict on data
    RULE: train_data's signature matches new_data's signature
    signature of data includes (namespace, input_feats, output_feats, type)
    """
    try:
        train_meta = read_data_meta(project_path = project_path, data_name = model_meta['train_data'])
    except Exception, e:
        raise e
        #raise RuntimeError('the model %s has NOT been trained on any data yet' % (model_meta['name'], ))
    signature_template = get_config('data_signature_template')
    train_sig = {sig:train_meta[sig] for sig in signature_template}
    data_sig = {sig:data_meta[sig] for sig in signature_template}
    ## rules to decide if data_sig is compatible with a model trained on data_sig
    ## data.namespace == train.namespace
    ## data.input_features >= train.input_features
    ## data.output_features == train.output_features - NO NEED FOR PREDICTION AT ALL
    ## data.type == train.type
    namespace_match = data_sig['namespace'] == train_sig['namespace']
    inputs_match = set(data_sig['input_features']).issuperset(set(train_sig['input_features']))
    #outputs_match = set(data_sig['output_features']) == set(train_sig['output_features'])
    type_match = data_sig['type'] == train_sig['type']
    compatible = namespace_match and inputs_match and type_match
    return compatible

def transformable(model_meta, data_meta, project_path = None):
    """to test if model can be used to transform on data (e.g. feature extractor/selector)
    RULE: train_data's signature matches new_data's signature
    signature of data includes (namespace, input_feats, output_feats, type)
    """
    try:
        train_meta = read_data_meta(project_path = project_path, data_name = model_meta['train_data'])
    except Exception, e:
        raise e
        #raise RuntimeError('the model %s has NOT been trained on any data yet' % (model_meta['name'], ))
    signature_template = get_config('data_signature_template')
    train_sig = {sig:train_meta[sig] for sig in signature_template}
    data_sig = {sig:data_meta[sig] for sig in signature_template}
    ## rules to decide if data_sig is compatible with a model trained on data_sig
    ## data.namespace == train.namespace
    ## data.input_features >= train.input_features
    ## data.output_features == train.output_features - NO NEED FOR PREDICTION AT ALL
    ## data.type == train.type
    namespace_match = data_sig['namespace'] == train_sig['namespace']
    inputs_match = set(data_sig['input_features']).issuperset(set(train_sig['input_features']))
    #outputs_match = set(data_sig['output_features']) == set(train_sig['output_features'])
    type_match = data_sig['type'] == train_sig['type']
    compatible = namespace_match and inputs_match and type_match
    return compatible

def train_meta_on(model_meta, data_meta, trained_model_name):
    """
    create and return trained_model_meta based on the previous model meta and data meta
    """
    trained_model_meta = cp.deepcopy(model_meta)
    train_data = data_meta['name']
    trained_model_meta.update({'name': trained_model_name, 'train_data': train_data})
    return trained_model_meta
    
def train_on(model_name, data_name, trained_model_name, project_path = None):
    """
    STEPS:
    1. load model_meta and data_meta if type DOESNT match, raise Exception
    2. load model_bulk and data_bulk into memory
    3. call model_bulk.fit(data_bulk)
    4. generate the newmodel and save it by trained_model_name
    """
    ## test if model is trainable on data
    model_meta = read_model_meta(project_path = project_path, model_name = model_name)
    data_meta = read_data_meta(project_path = project_path, data_name = data_name)
    if not trainable(model_meta, data_meta):
        raise RuntimeError("model %s is not trainable on dataset %s" % (model_name, data_name))
    ## load into memory
    model_bulk = read_model_bulk(project_path = project_path, model_name = model_name)
    data_bulk = read_data_bulk(project_path = project_path, data_name = data_name)
    ## call model.fit(data)
    input_feats = data_meta['input_features']
    output_feats = data_meta['output_features']
    if len(output_feats) == 1:
        output_feats = output_feats[0]
    data_input = np.asarray(data_bulk.loc[:, input_feats])
    data_output = np.asarray(data_bulk.loc[:, output_feats])
    model_bulk.fit(data_input, data_output)
    ## generate new model
    trained_model_meta = train_meta_on(model_meta, data_meta, trained_model_name)
    write_model(project_path = project_path, model_meta = trained_model_meta, model_bulk = model_bulk)
    
def predict_on(model_name, data_name, predicted_data_name, project_path = None):
    """
    STEP:
    1. trace model train_data's meta
    2. compare train_data signature with new_data signature to see if they match
    3. signature defined as namespace/namespace1
    """
    ## read meta information
    model_meta = read_model_meta(project_path = project_path, model_name = model_name)
    model_train_meta = read_data_meta(project_path = project_path, data_name = model_meta['train_data'])
    data_meta = read_data_meta(project_path = project_path, data_name = data_name)
    if not predictable(model_meta, data_meta, project_path):
        raise RuntimeError("model %s cannot predict on data %s" % (model_name, data_name))
    ## load the data and trained model into memory
    model_bulk = read_model_bulk(project_path = project_path, model_name = model_name)
    data_bulk = read_data_bulk(project_path = project_path, data_name = data_name)
    input_features = model_train_meta['input_features']
    output_features = model_train_meta['output_features']
    ## fit the data into the shape of model's train data
    X = np.asarray(data_bulk.loc[:, input_features])
    yhat = model_bulk.predict(X)
    ## combine yhat with original data
    yhat = pd.DataFrame(yhat, columns = output_features)
    predicted_data_bulk = data_bulk
    predicted_data_bulk.update(yhat, join = 'left')
    predicted_data_meta = data_meta
    predicted_data_meta.update({'name': predicted_data_name, 'output_features': output_features})
    write_data(project_path = project_path, data_meta = predicted_data_meta, data_bulk = predicted_data_bulk)

def transform_on(model_name, data_name, transformed_data_name, project_path = None):
    """
    """
    model_meta = read_model_meta(project_path = project_path, model_name = model_name)
    model_train_meta = read_data_meta(project_path = project_path, data_name = model_meta['train_data'])
    data_meta = read_data_meta(project_path = project_path, data_name = data_name)
    if not transformable(model_meta, data_meta, project_path):
        raise RuntimeError("model %s cannot transform on data %s" % (model_name, data_name))

    model_bulk = read_model_bulk(project_path = project_path, model_name = model_name)
    data_bulk = read_data_bulk(project_path = project_path, data_name = data_name)
    input_features = model_train_meta['input_features']
    output_features = model_train_meta['output_features']

    X = np.asarray(data_bulk.loc[:, input_features])
    yhat = model_bulk.transform(X)
    ## combine yhat with original data
    n_ycols = yhat.shape[1]
    transformed_features = ['%s_%i' % (model_name, i) for i in xrange(n_ycols)]
    yhat = pd.DataFrame(yhat, columns = transformed_features)
    predicted_data_bulk = data_bulk
    #predicted_data_bulk.update(yhat, join = 'left')
    #predicted_data_bulk = hglue([data_bulk, yhat])
    ## excluding original input features
    predicted_data_bulk = hglue([yhat, data_bulk.loc[:, output_features]])
    predicted_data_meta = data_meta
    predicted_data_meta.update({'name': transformed_data_name, 'input_features': transformed_features})
    write_data(project_path = project_path, data_meta = predicted_data_meta, data_bulk = predicted_data_bulk)

def score_on(target_data_name, predicted_data_name, score_fn, project_path = None):
    """
    the target_data and predicted_data should have the common set of output feature
    the method will compare the output features and output and apply score_fn on them
    examples of score_fn include: (1) sklearn.metrics.XX
    TODO: consider the type of data to test if the certain score_fn is appliable to them
    """
    ## read meta
    target_meta = read_data_meta(project_path = project_path, data_name = target_data_name)
    predicted_meta = read_data_meta(project_path = project_path, data_name = predicted_data_name)
    assert target_meta['output_features'] == predicted_meta['output_features'] 
    output_features = target_meta['output_features']
    ## read bulk
    y = np.asarray(read_data_bulk(project_path = project_path, data_name = target_data_name).loc[:, output_features])
    yhat = np.asarray(read_data_bulk(project_path = project_path, data_name = predicted_data_name).loc[:, output_features])
    ## apply scorefn 
    return score_fn(y, yhat)

def set_params_on_model(model_name, model_params, project_path):
    """
    params: {param_name: param_value}
    """
    ## find model meta and bulk
    model_meta = read_model_meta(model_name, project_path)
    model_bulk = read_model_bulk(model_name, project_path)
    ## set model params on bulk
    ## model meta will not be changed
    model_bulk.set_params(**model_params)
    ## save it back
    write_model(model_meta, model_bulk, project_path)

## pipeline 

def make_pipeline(pipeline_name):
    pipeline = nx.DiGraph()
    pipeline.name = pipeline_name
    #pipeline.max_id = 0 ## infinite source of id
    return pipeline

def make_pipe(fn, pipeline, pipe_name):
    #pipe_id = pipeline.max_id
    #pipeline.max_id += 1 
    arg_names = inspect.getargspec(fn).args
    pipe_config = {'name': pipe_name, 'fn': fn, 'args': arg_names, 
                    'bindings': {k:None for k in arg_names}}
    pipeline.add_node(pipe_name, pipe_config)
    return pipe_name

def connect_pipes(from_id, out_arg, to_id, in_arg, pipeline):
    pipeline.add_edge(from_id, to_id)
    from_node = pipeline.node[from_id]
    to_node = pipeline.node[to_id]
    #assert from_node['bindings'][out_arg] is None
    #assert to_node['bindings'][in_arg] is None
    #common_item = '%s_(%s_TO_%s)' % (pipeline.name, from_node['name'], to_node['name'])
    ## because one output might flow into multiple inputs of different nodes
    common_item = '%s_(%s_OUT)' % (pipeline.name, from_node['name'])
    from_node['bindings'][out_arg] = common_item
    to_node['bindings'][in_arg] = common_item
    return pipeline

def bind_args(args, values, pipe_id, pipeline):
    pipe_node = pipeline.node[pipe_id]
    for arg, value in zip(args, values):
        assert pipe_node['bindings'][arg] is None
        pipe_node['bindings'][arg] = value
        
def run_pipeline(pipeline):
    for pipe_id in nx.topological_sort(pipeline):
        pipe_node = pipeline.node[pipe_id]
        fn = pipe_node['fn']
        bindings = pipe_node['bindings']
        fn(**bindings)

def runnable_pipeline(pipeline):
    def _pipeline():
        run_pipeline(pipeline)
    return _pipeline

def inspect_pipeline(pipeline):
    figure(figsize=(12, 12))
    layout = nx.graphviz_layout(pipeline)
    nx.draw(pipeline, pos = layout)

## parameter optimization

def random_param_optimize(models2params, n_iter, fn0, scorefn0, project_path):
    """
    models2params: {modelname: {paramname: paramvalue, ...}, ...}
    n_iter: number of tries
    fn = zero_arg function that runs and uses models or datafiles, 
        e.g., partial(train_on), runnable_pipeline instance and etc.
    scorefn0: zero_arg function that tells how good the performance of each run is, 
        e.g., partial(score_on, metric.XXX) 
    RETURN: [(setting1, score1), ...], setting = {(model_name, param_name): param_value}
    """
    ## group parameters
    flat_params = {}
    for model_name, params in models2params.items():
        for param_name,  param_values in params.items():
            flat_param_name = (model_name, param_name)
            flat_params[flat_param_name] = param_values
    ## sampling parameter space
    param_sampler = grid_search.ParameterSampler(flat_params, n_iter)

    results = []
    ## iterate each param setting
    for setting in param_sampler:
        ## get params for each model
        models2settings = defaultdict(dict) 
        for (model_name, param_name), param_value in setting.items():
            models2settings[model_name][param_name] = param_value
        ## set params on each model
        for model_name, params in models2settings.items():
            set_params_on_model(model_name, params, project_path)
        ## run fn 
        fn0() 
        ## record score 
        score = scorefn0()
        results.append((setting, score)) 
    ## sort the results based on score:
    results = sorted(results, key = lambda (setting, score): score, reverse = True)
    return results 