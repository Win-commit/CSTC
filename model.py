import os
os.environ["DGLBACKEND"] = "pytorch"
import sys
sys.path.append('./HGP-SL') 
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
from layers import ConvPoolReadout
import dgl

class Finsentry(nn.Module):
    def __init__(self, in_dim:list, hidden_dim:list, out_dim,n_layers=3):
        super(Finsentry, self).__init__()
        self.num_layers=n_layers
        Stock_convpools=[] 
        Guarantee_convpools=[] 
        Transaction_convpools=[]
        for i in range(n_layers):
            if i==0:
                Stock_convpools.append(ConvPoolReadout(in_dim[0], hidden_dim[0]))
                Guarantee_convpools.append(ConvPoolReadout(in_dim[1], hidden_dim[1]))
                Transaction_convpools.append(ConvPoolReadout(in_dim[2], hidden_dim[2]))
            else:
                Stock_convpools.append(ConvPoolReadout(hidden_dim[0], hidden_dim[0]))
                Guarantee_convpools.append(ConvPoolReadout(hidden_dim[1], hidden_dim[1]))
                Transaction_convpools.append(ConvPoolReadout(hidden_dim[2], hidden_dim[2]))
        #GCNandPOOL   
        self.Stock_convpools = nn.ModuleList(Stock_convpools)
        self.Guarantee_convpools = nn.ModuleList(Guarantee_convpools)
        self.Transaction_convpools = nn.ModuleList(Transaction_convpools) 

        #Convert readout
        self.t2g=nn.Linear(2*hidden_dim[2],hidden_dim[1])  
        self.g2s=nn.Linear(2*hidden_dim[1],hidden_dim[0])
        self.s2t=nn.Linear(2*hidden_dim[0],hidden_dim[2])
        self.s2g=nn.Linear(2*hidden_dim[0],hidden_dim[1])
        self.g2t=nn.Linear(2*hidden_dim[1],hidden_dim[2])
        self.t2s=nn.Linear(2*hidden_dim[2],hidden_dim[0])
        
        #GATLayer
        self.gatS=GATConv(hidden_dim[0],hidden_dim[0],num_heads=1,allow_zero_in_degree=True)
        self.gatG=GATConv(hidden_dim[1],hidden_dim[1],num_heads=1,allow_zero_in_degree=True)
        self.gatT=GATConv(hidden_dim[2],hidden_dim[2],num_heads=1,allow_zero_in_degree=True)

        #Classifier
        lin_output=hidden_dim[0]+hidden_dim[1]+hidden_dim[2]
        self.lin1=nn.Linear(2*lin_output,lin_output)
        self.lin2=nn.Linear(lin_output,lin_output//2)
        self.lin3=nn.Linear(lin_output//2,out_dim)

       

    def forward(self, batched_graphs):
        batched_graphs[0]=dgl.add_self_loop(batched_graphs[0])
        batched_graphs[1]=dgl.add_self_loop(batched_graphs[1])
        batched_graphs[2]=dgl.add_self_loop(batched_graphs[2])


        stock=batched_graphs[0]
        guarantee= batched_graphs[1]
        transaction= batched_graphs[2]



        for i in range(self.num_layers-1):
            #Stock
            stock=dgl.add_self_loop(stock)
            stock,sndata,_,sreadout=self.Stock_convpools[i](stock,stock.ndata['h'])
            #Guarantee
            guarantee=dgl.add_self_loop(guarantee)
            guarantee,gndata,_,greadout=self.Guarantee_convpools[i](guarantee,guarantee.ndata['h'])
            #Transaction
            transaction=dgl.add_self_loop(transaction)
            transaction,tndata,_,treadout=self.Transaction_convpools[i](transaction,transaction.ndata['h'])

            if i%2==0:
                #Convert readout
                guarantee_supernode_f=F.relu(self.t2g(treadout))
                stock_supernode_f=F.relu(self.g2s(greadout))
                transaction_supernode_f=F.relu(self.s2t(sreadout))
            else:
                guarantee_supernode_f=F.relu(self.s2g(sreadout))
                transaction_supernode_f=F.relu(self.g2t(greadout))
                stock_supernode_f=F.relu(self.t2s(treadout))
            
            graphs=[stock,guarantee,transaction]
            for i in range(3):
                G=graphs[i]
                if i==0:
                    new_node_feature=stock_supernode_f
                    G.ndata['h']=sndata
                    gat_layer=self.gatS
                elif i==1:
                    new_node_feature=guarantee_supernode_f
                    G.ndata['h']=gndata
                    gat_layer=self.gatG
                else:
                    new_node_feature=transaction_supernode_f
                    G.ndata['h']=tndata
                    gat_layer=self.gatT
                G_copy=G.clone()
                with G_copy.local_scope():
                    G_copy.add_nodes(1)
                    new_node_id = G_copy.number_of_nodes() - 1
                    G_copy.add_edges(new_node_id, torch.arange(0, new_node_id))
                    G_copy.ndata['h'][new_node_id] =new_node_feature
                    new_features = gat_layer(G_copy, G_copy.ndata['h']).squeeze(1)
                    G_copy.ndata['h'][:new_node_id] = new_features[:new_node_id]
                
                G.ndata['h'] = G_copy.ndata['h'][:G.number_of_nodes()]

        
        _,_,_,sreadout=self.Stock_convpools[i](stock,stock.ndata['h'])     
        _,_,_,greadout=self.Guarantee_convpools[i](guarantee,guarantee.ndata['h'])
        _,_,_,treadout=self.Transaction_convpools[i](transaction,transaction.ndata['h'])

        #Classifier
        n_feat=torch.cat((sreadout,greadout,treadout),dim=-1)
        n_feat=F.relu(self.lin1(n_feat))
        n_feat=F.relu(self.lin2(n_feat))
        n_feat=self.lin3(n_feat)

        return F.log_softmax(n_feat,dim=-1)

                
                