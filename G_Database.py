import os
import pandas as pd
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
import numpy as np
import torch.nn.functional as F
import ast


class CompanyDataset(DGLDataset):
    def __init__(self,meta_path='./Meta/metadata.csv'):
        self.meta = pd.read_csv(meta_path, index_col=0)
        super().__init__(name="Company")
        
    

    def process(self):
        self.graphs = []
        self.labels = []

        stock_graphs=pd.read_csv('./Stock/graph_edges.csv')
        stock_edges_group=stock_graphs.groupby('graph_id')
        stock_features=pd.read_csv('./Stock/features.csv',encoding='gbk')

        guarantee_graphs=pd.read_csv('./Guarantee/graph_edges.csv')
        guarantee_edges_group=guarantee_graphs.groupby('graph_id')
        guarantee_features=pd.read_csv('./Guarantee/features.csv',encoding='gbk')

        transaction_graphs=pd.read_csv('./Transaction/graph_edges.csv')
        transaction_edges_group=transaction_graphs.groupby('graph_id')
        transaction_features=pd.read_csv('./Transaction/features.csv',encoding='gbk')

        for _, row in self.meta.iterrows():
            self.labels.append(row['label'])

            #stock图和特征
            stock_id=row['stock']
            stock_graph = self.build_graph(stock_id, stock_edges_group, stock_features,'stock')
            
            #guarantee图和特征
            guarantee_id=row['guarantee']
            guarantee_graph = self.build_graph(guarantee_id, guarantee_edges_group, guarantee_features,'guarantee')

            #transaction图和特征
            transaction_id=row['transaction']
            transaction_graph = self.build_graph(transaction_id, transaction_edges_group, transaction_features,'transaction')

            self.graphs.append([stock_graph, guarantee_graph, transaction_graph])
            
        
    @classmethod
    def build_graph(cls,id, edges_group, features,types):
        edges_of_id = edges_group.get_group(id)

        #将元组字符串转变为元组
        edges_of_id['src'] =edges_of_id['src'].apply(ast.literal_eval)
        edges_of_id['dst'] =edges_of_id['dst'].apply(ast.literal_eval)

        edges_of_id[['src_node_id', 'src_map_id']] = edges_of_id['src'].apply(pd.Series)
        edges_of_id[['dst_node_id', 'dst_map_id']] = edges_of_id['dst'].apply(pd.Series)

        src=edges_of_id['src_node_id'].to_numpy()
        dst=edges_of_id['dst_node_id'].to_numpy()

        src_dict = edges_of_id.set_index('src_node_id')['src_map_id'].to_dict()
        dst_dict = edges_of_id.set_index('dst_node_id')['dst_map_id'].to_dict()
        g = dgl.DGLGraph((src, dst))
        g = dgl.add_self_loop(g)
        if types!='stock':
            g=dgl.to_bidirected(g)
        
        # add node features
        dimension = features.shape[1]-1
        g.ndata['h']=torch.ones((g.num_nodes(),dimension))
        for i in range(g.num_nodes()):
            if i in src_dict:
                if types=='stock':
                    feature = features[features['ID']==src_dict[i]][['风险集中','股权关联度','股权集中度']].to_numpy()
                elif types=='guarantee':
                    feature = features[features['ID']==src_dict[i]][['担保总额','担保关联度','担保交易']].to_numpy()
                elif types=='transaction':
                    feature = features[features['ID']==src_dict[i]][['大额交易','交易广度']].to_numpy()
            elif i in dst_dict:
                if types=='stock':
                    feature = features[features['ID']==dst_dict[i]][['风险集中','股权关联度','股权集中度']].to_numpy()
                elif types=='guarantee':
                    feature = features[features['ID']==dst_dict[i]][['担保总额','担保关联度','担保交易']].to_numpy()
                elif types=='transaction':
                    feature = features[features['ID']==dst_dict[i]][['大额交易','交易广度']].to_numpy()
            if feature.shape[0]==1:
                feature = feature.astype(np.float32)
                g.ndata['h'][i] = torch.tensor(feature)
            else:
                g.ndata['h'][i] = torch.randn(1, dimension)
            g.ndata['h']=F.normalize(g.ndata['h'], p=2, dim=0)
        return g


        

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
    

