"""
Authors: Nick Gisolfi, Dan Howarth
"""
from json import dumps
from re import split as regex_split

class ARFReader:
    """Reads autonRF model file into sklearn random forest. This will only load the tree structure. The metadata for
    the sklearn random forest will be all wiggidy-wack.

    Currently this is only used to test if the outputs of python version for inbounds, mean entropy, etc match the
    outputs of the C version.

    Requires sklearn-json. Use  fork at https://github.com/howarth/sklearn-json/ for a version compatible with more
    recent sklearn files.

    usage:
        arf_reader = ARFReader(autonRF_c_model_file)
        arf_reader.write2json(temporary_json_file)
        rf_model = sklearn_json.from_json(temporary_json_file)

    """

    UNALLOWED_CHARS = '!@#$%^&*()_+~`=\-\[\]\{\}:;\"\'<>,.\\/?|'

    def __init__( self, fileName ):
        with open(fileName,'r') as fp:
            self.params = self.read_params(fp)
            self.model = self.read_model(fp)

    def make_safe_names( self, names ):
        return [''.join([word.capitalize() for word in regex_split('['+self.UNALLOWED_CHARS+']+',attName)]) for attName in names]

    def read_params( self, fp ):
        params={}

        #Skip ARF Preamble
        assert fp.readline().strip()=='<bag_model>'
        assert fp.readline().strip()=='1'
        assert fp.readline().strip()=='<string_array>'
        assert fp.readline().strip()=='size 1'
        fp.readline().strip()
        assert fp.readline().strip()=='</string_array>'

        #Read Attribute Names
        assert fp.readline().strip()=='<string_array>'
        params['num_atts']=int(fp.readline().strip().split()[1])
        params['att_names'] = self.make_safe_names( [fp.readline().strip() for attnum in range(params['num_atts'])] )
        assert fp.readline().strip()=='</string_array>'

        #Read #Trees and Target Attribute Number
        params['num_trees']=int(fp.readline().strip())
        params['target_attnum']=int(fp.readline().strip())

        #Read Class Names
        assert fp.readline().strip()=='<string_array>'
        params['num_classes'] = int(fp.readline().strip().split()[1])
        params['class_names'] = [fp.readline().strip() for classnum in range(params['num_classes'])]
        assert fp.readline().strip()=='</string_array>'

        return params

    def read_model( self, fp ):
        model={}
        nID = 0

        #Read Trees
        for mID in range(1,self.params['num_trees']+1):
            assert fp.readline().strip()=='<decision_tree>'
            tree,nodeID = self.mk_node( fp, nID, mID )
            model[mID] = tree
            assert fp.readline().strip()=='</decision_tree>'
        return model

    def mk_node( self, fp, nID, mID ):
        node={'nID':nID}
        node['mID']=mID
        assert fp.readline().strip()=='<decision_node>'
        #Is this a leaf?
        if fp.readline().strip()=='true':
            node['is_leaf']=True
            #is this leaf a classification leaf?
            if fp.readline().strip() == 'true':
                node['is_classification']=True
                #read sample counts at this leaf
                assert fp.readline().strip()=='<dyv>'
                numElements=int(fp.readline().strip().split()[1])
                node['distribution']=[int(fp.readline().strip()) for eID in range(numElements)]
                node['nodes_json']=[-1,
                                    -1,
                                    -2,
                                    -2.0,
                                    0,
                                    0,
                                    0
                                   ]
                node['values_json']=[[float(e) for e in node['distribution']]]
                assert fp.readline().strip()=='</dyv>'
                assert fp.readline().strip()=='</decision_node>'
            else:
                print('Regression Not Implemented')
                #read mean output of the node NOT IMPLEMENTED
                pass
        else:
            #this is not a leaf
            node['is_leaf']=False
            node['att_num']=int(fp.readline().strip())
            node['att_name']=self.params['att_names'][node['att_num']]
            node['is_symbolic']=fp.readline().strip()
            node['contains_missing_values']=fp.readline().strip()
            node['missing_value_decision_path']=fp.readline().strip()
            node['threshold']=float(fp.readline().strip())
            node['min']=fp.readline().strip()
            node['max']=fp.readline().strip()
            node['left_child_nID']=nID+1
            LC,nID=self.mk_node(fp,nID+1,mID)
            node['right_child_nID']=nID+1
            RC,nID=self.mk_node(fp,nID+1,mID)
            node['nodes_json']=[node['left_child_nID'],
                          node['right_child_nID'],
                          node['att_num'],
                          node['threshold'],
                          0, #gini or whatever
                          0, #number of samples in leaf
                          0 #weighted sum of samples in leaf
                         ]
            node['values_json']=[[0.0 for i in range(self.params['num_classes'])]]
            node['left_child']=LC
            node['right_child']=RC
            assert fp.readline().strip()=='</decision_node>'

        return node,nID

    # It wound up being easiear to read in ARF, rewrite to file as JSON with scikit-learn-esque structure, then read back in
    def write2json( self, filename ):
        with open(filename, 'w') as writer:
            writer.write(dumps(self.to_json()))

    def nodes2json( self, json, node ):
        json.append(node['nodes_json'])
        if not node['is_leaf']:
            json = self.nodes2json(json,node['left_child'])
            json = self.nodes2json(json,node['right_child'])
        return json

    def values2json( self, json, node ):
        json.append(node['values_json'])
        if not node['is_leaf']:
            json = self.values2json(json,node['left_child'])
            json = self.values2json(json,node['right_child'])
        return json

    def dtypes2json( self, json, node ):
        return ['<i8','<i8','<i8','<f8','<f8','<i8','<f8']

    def get_node_count( self, count, node ):
        count +=1
        if not node['is_leaf']:
            count = self.get_node_count(count,node['left_child'])
            count = self.get_node_count(count,node['right_child'])
        return count

    def to_json( self ):
        rf = {'meta':'rf',
              'max_depth':0,
              'min_samples_split':2,
              'min_samples_leaf':1,
              'min_weight_fraction_leaf':0.0,
              'max_features':'auto',
              'max_leaf_nodes':None,
              'min_impurity_decrease':0.0,
              'n_features_': self.params['num_atts'],
              'n_outputs_':1,
              'classes_': [i for i in range(self.params['num_classes'])],
              'estimators_': [self.estimator2json(i) for i in range(1,self.params['num_trees']+1)],
              'params':{'bootstrap':True,
                        'class_weight':None,
                        'criterion':'ALRF',
                        'max_depth':0,
                        'max_features':None,
                        'max_leaf_nodes':None,
                        'min_impurity_decrease':0.0,
                        'min_samples_leaf':1,
                        'min_samples_split':2,
                        'min_weight_fraction_leaf':0.0,
                        'n_estimators': self.params['num_trees'],
                        'n_jobs':None,
                        'oob_score':None,
                        'random_state':0,
                        'verbose':0,
                        'warm_start':False
                       },
              'n_classes_':self.params['num_classes']
             }
        return rf

    def estimator2json( self, mID ):
        dt = {'meta':'decision-tree',
              'feature_importances_':[0 for i in range(self.params['num_atts']-1)],
              'max_features_':self.params['num_atts']-1,
              'n_classes_':self.params['num_classes'],
              'n_features_':self.params['num_atts']-1,
              'n_outputs_':1,
              'tree_':{'max_depth':0,
                       'node_count':self.get_node_count(0,self.model[mID]),
                       'nodes':self.nodes2json([],self.model[mID]),
                       'values':self.values2json([],self.model[mID]),
                       'nodes_dtype':self.dtypes2json([],self.model[mID])
                      },
              'classes_':[0, 1, 2],
              'params':{'class_weight':None,
                        'criterion':'ALRF',
                        'max_depth':0,
                        'max_features':'auto',
                        'max_leaf_nodes':None,
                        'min_impurity_decrease':0.0,
                        'min_samples_leaf':1,
                        'min_samples_split':2,
                        'min_weight_fraction_leaf':0.0,
                        #'presort':False,
                        'random_state':15599,
                        'splitter':'best'
                       }
             }
        return dt
