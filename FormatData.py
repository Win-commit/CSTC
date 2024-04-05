import pandas as pd
import os

'''
这里只用来构建担保，关联交易，股权的图拓扑格式，不负责处理节点与边的特征
在担保/关联交易/股权处理中，会有一个映射文件:: 公司全称:(center_id?,map_id),即是否为中心节点,以及映射id
也会有一个Graph edges, 用于描述每个中心企业图的图topo结构,格式为Graph_id source_id target_id。
src是一个元组,形如(nodeid,mapid),表示节点id和映射id.在每张新图中,节点id都是由0开始递增的
这里的Graph_id是指每个中心企业的id, source_id是指源节点的id, target_id是指目标节点的id
也会有一个Graph properties文件,用于描述每个中心企业图的图的属性,格式为Graph_id, label, num_nodes
'''

'''
详情可见：
https://docs.dgl.ai/tutorials/blitz/6_load_data.html#sphx-glr-tutorials-blitz-6-load-data-py
'''

class GuaranteeData:

    def __init__(self, excel_path):
        self.path = excel_path

    def read_data(self):
        # 读取数据
        df = pd.read_excel(self.path,index_col=0)
        #删除以个人身份作为担保方的数据
        df = df[df['担保方公司名称'].str.contains('有限公司')]
        # 选择需要的列
        self.data= df[['担保方公司名称', '被担保方公司名称']].reset_index(drop=True)


    def nameToid(self):
        # 生成公司名称到id的映射
        unique_body_names = self.data['担保方公司名称'].unique()
        name_to_id = {name: (True, i) for i, name in enumerate(unique_body_names)}
        unique_names = self.data['被担保方公司名称'].unique()

        current_max_id = max(name_to_id.values(), key=lambda x: x[1])[1]
        
        for name in unique_names:
            if name not in name_to_id:
                current_max_id += 1
                name_to_id[name] = (False, current_max_id)

        self.name_to_id = name_to_id

        df = pd.DataFrame.from_dict(name_to_id, orient='index', columns=['IsCenter', 'ID'])
        df = df.reset_index().rename(columns={'index': 'Fullname'})
        
        # 将DataFrame保存为CSV文件
        df.to_csv('Guarantee/mapped.csv')


    def generateGraphEdges(self):
        # 创建一个空的DataFrame，列名为'graph_id', 'src', 'dst'
        edges = pd.DataFrame(columns=['graph_id', 'src', 'dst'])

        # 创建一个字典来存储每个图的当前节点计数器
        node_counters = {}

        # 遍历data DataFrame的每一行
        for index, row in self.data.iterrows():
            # 获取'担保方公司名称'和'被担保方公司名称'的map_id
            guarantor_map_id = self.name_to_id[row['担保方公司名称']][1]
            body_map_id = self.name_to_id[row['被担保方公司名称']][1]

            # 如果这是图的第一个节点，初始化节点计数器
            if guarantor_map_id not in node_counters:
                node_counters[guarantor_map_id] = 1

            # 添加一行到edges DataFrame
            '''
            根据这个要建立无向图,危机是可以双向传播的
            '''
            edges = edges._append({
                'graph_id': guarantor_map_id,
                'src': (0, guarantor_map_id),
                'dst': (node_counters[guarantor_map_id], body_map_id)
            }, ignore_index=True)


            # 更新节点计数器
            node_counters[guarantor_map_id] += 1

        # 将edges DataFrame保存为CSV文件
        edges.to_csv('Guarantee/graph_edges.csv', index=False)
        
        node_counts = pd.DataFrame.from_dict(node_counters, orient='index', columns=['node_count'])
        node_counts = node_counts.reset_index().rename(columns={'index': 'graph_id'})
        
        # 将node_counts DataFrame保存为CSV文件
        node_counts.to_csv('Guarantee/node_counts.csv')

    def pipline(self):
        if not os.path.exists('Guarantee'):
            os.makedirs('Guarantee')
        
        self.read_data()
        self.nameToid()
        self.generateGraphEdges()




class StockData:

    def __init__(self, excel_paths):
        '''
        第一个地址是股东信息表，后面的是控股企业表
        '''
        self.pathlist = excel_paths
        


    def read_data(self):
        # 读取数据
        self.datals = []

        df = pd.read_excel(self.pathlist[0],index_col=0)
        col_ls=['企业名称']
        for i in range(1,11):
            col_ls.append(f'大股东名称第{i}名')
        self.datals.append(df[col_ls].reset_index(drop=True))
        

        for path in self.pathlist[1:]:
            df = pd.read_excel(path,index_col=0)
            # 选择需要的列
            self.datals.append(df[['企业名称', '控股企业名称']].reset_index(drop=True))


    def nameToid(self):
        # 生成公司名称到id的映射
        name_to_id = {}
        for index in range(1,len(self.datals)):
            data = self.datals[index]
            unique_body_names = data['企业名称'].unique()
            if not name_to_id:
                name_to_id = {name: (True, i) for i, name in enumerate(unique_body_names)}
            else:
                current_max_id = max(name_to_id.values(), key=lambda x: x[1])[1]
                for name in unique_body_names:
                    if name not in name_to_id:
                        current_max_id += 1
                        name_to_id[name] = (False, current_max_id)

            unique_names = data['控股企业名称'].unique()
            current_max_id = max(name_to_id.values(), key=lambda x: x[1])[1]
        
            for name in unique_names:
                if name not in name_to_id:
                    current_max_id += 1
                    name_to_id[name] = (False, current_max_id)

        

        #股东
        data=self.datals[0]

        col_ls=[]
        current_max_id = max(name_to_id.values(), key=lambda x: x[1])[1]

        for i in range(1,11):
            col_ls.append(f'大股东名称第{i}名')

        for col in col_ls:
            shareholder=data[col].unique()
            for name in shareholder:
                if name not in name_to_id:
                    current_max_id += 1
                    name_to_id[name] = (False, current_max_id)

        self.name_to_id = name_to_id
        df = pd.DataFrame.from_dict(name_to_id, orient='index', columns=['IsCenter', 'ID'])
        df = df.reset_index().rename(columns={'index': 'Fullname'})
        
        # 将DataFrame保存为CSV文件
        df.to_csv('Stock/mapped.csv')



    def generateGraphEdges(self):
        # 创建一个空的DataFrame，列名为'graph_id', 'src', 'dst'
        edges = pd.DataFrame(columns=['graph_id', 'src', 'dst'])

        # 创建一个字典来存储每个图的当前节点计数器
        node_counters = {}

        # 广度优先遍历
        for i in range(1,len(self.datals)):
            data = self.datals[i]

            for _, row in data.iterrows():
                # 获取'担保方公司名称'和'被担保方公司名称'的map_id
                if i==1:
                    center_map_id = self.name_to_id[row['企业名称']][1]
                else:
                    src_map_id = self.name_to_id[row['企业名称']][1]
                    df=edges.copy()
                    df['dst_map_id'] = df['dst'].apply(lambda x: x[1])
                    filtered_df = df[(df['dst_map_id'] == src_map_id)]
                    graph_ids = filtered_df['graph_id'].unique()

                body_map_id = self.name_to_id[row['控股企业名称']][1]

                # 如果这是图的第一个节点，初始化节点计数器
                if i==1 and (center_map_id not in node_counters) :
                    node_counters[center_map_id] = 1

                # 添加一行到edges DataFrame

                if i==1:
                    edges = edges._append({
                        'graph_id': center_map_id,
                        'src': (0, center_map_id),
                        'dst': (node_counters[center_map_id], body_map_id)
                        }, ignore_index=True)
                    # 更新节点计数器
                    node_counters[center_map_id] += 1
                else:
                    for center_map_id in graph_ids:
                        filtered_df = df[df['graph_id'] == center_map_id]
                        node_id_src = filtered_df[filtered_df['dst'].apply(lambda x: x[1]) == src_map_id]['dst'].apply(lambda x: x[0]).unique()[0]
                        node_id_dst = filtered_df[filtered_df['dst'].apply(lambda x: x[1]) == body_map_id]['dst'].apply(lambda x: x[0]).unique()
                        if len(node_id_dst)==0:
                            node_id_dst=node_counters[center_map_id]
                            node_counters[center_map_id] += 1
                        else:
                            node_id_dst=node_id_dst[0]
                        edges = edges._append({
                            'graph_id': center_map_id,
                            'src': (node_id_src, src_map_id),
                            'dst': (node_id_dst, body_map_id)
                            }, ignore_index=True)


        shareholders=self.datals[0]
        for _, row in shareholders.iterrows():
            center_map_id = self.name_to_id[row['企业名称']][1]

            for i in range(1,11):
                shareholder=row[f'大股东名称第{i}名']
                if shareholder in self.name_to_id:
                    shareholder_map_id=self.name_to_id[shareholder][1]
                    df=edges.copy()
                    df['dst_map_id'] = df['dst'].apply(lambda x: x[1])
                    node_id_src = df[(df['dst_map_id'] == shareholder_map_id)]['dst'].apply(lambda x: x[0]).unique()
                    if len(node_id_src)==0:
                        node_id_src=node_counters[center_map_id]
                        node_counters[center_map_id] += 1
                    else:
                        node_id_src=node_id_src[0]
                    edges = edges._append({
                        'graph_id': center_map_id,
                        'src': (node_id_src, shareholder_map_id),
                        'dst': (0, center_map_id)
                        }, ignore_index=True)


        # 将edges DataFrame保存为CSV文件
        edges = edges.sort_values('graph_id', ascending=True)    
        edges.to_csv('Stock/graph_edges.csv', index=False)
        
        node_counts = pd.DataFrame.from_dict(node_counters, orient='index', columns=['node_count'])
        node_counts = node_counts.reset_index().rename(columns={'index': 'graph_id'})
        
        # 将node_counts DataFrame保存为CSV文件
        node_counts.to_csv('Stock/node_counts.csv')

    def pipline(self):
        if not os.path.exists('Stock'):
            os.makedirs('Stock')
        
        self.read_data()
        self.nameToid()
        self.generateGraphEdges()



class TransactionData:

    def __init__(self, excel_path):
        self.path = excel_path

    def read_data(self):
        # 读取数据
        df = pd.read_excel(self.path,index_col=0)
        # 选择需要的列
        self.data= df[['公司名称', '关联方']].reset_index(drop=True)


    def nameToid(self):
        # 生成公司名称到id的映射
        unique_body_names = self.data['公司名称'].unique()
        name_to_id = {name: (True, i) for i, name in enumerate(unique_body_names)}
        unique_names = self.data['关联方'].unique()

        current_max_id = max(name_to_id.values(), key=lambda x: x[1])[1]
        
        for name in unique_names:
            if name not in name_to_id:
                current_max_id += 1
                name_to_id[name] = (False, current_max_id)

        self.name_to_id = name_to_id

        df = pd.DataFrame.from_dict(name_to_id, orient='index', columns=['IsCenter', 'ID'])
        df = df.reset_index().rename(columns={'index': 'Fullname'})
        
        # 将DataFrame保存为CSV文件
        df.to_csv('Transaction/mapped.csv')


    def generateGraphEdges(self):
        # 创建一个空的DataFrame，列名为'graph_id', 'src', 'dst'
        edges = pd.DataFrame(columns=['graph_id', 'src', 'dst'])

        # 创建一个字典来存储每个图的当前节点计数器
        node_counters = {}

        # 遍历data DataFrame的每一行
        for index, row in self.data.iterrows():
            # 获取'公司名称'和'关联方'的map_id
            center_map_id = self.name_to_id[row['公司名称']][1]
            body_map_id = self.name_to_id[row['关联方']][1]

            # 如果这是图的第一个节点，初始化节点计数器
            if center_map_id not in node_counters:
                node_counters[center_map_id] = 1

            # 添加一行到edges DataFrame
            '''
            根据这个要建立无向图,交易是双向的
            '''
            edges = edges._append({
                'graph_id': center_map_id,
                'src': (0, center_map_id),
                'dst': (node_counters[center_map_id], body_map_id)
            }, ignore_index=True)


            # 更新节点计数器
            node_counters[center_map_id] += 1

        # 将edges DataFrame保存为CSV文件
        edges.to_csv('Transaction/graph_edges.csv', index=False)
        
        node_counts = pd.DataFrame.from_dict(node_counters, orient='index', columns=['node_count'])
        node_counts = node_counts.reset_index().rename(columns={'index': 'graph_id'})
        
        # 将node_counts DataFrame保存为CSV文件
        node_counts.to_csv('Transaction/node_counts.csv')

    def pipline(self):
        if not os.path.exists('Transaction'):
            os.makedirs('Transaction')
        
        self.read_data()
        self.nameToid()
        self.generateGraphEdges()



if __name__ == '__main__':
    paths=['./Info Sheet/股东信息.xlsx','./Info Sheet/stock0-1.xlsx','./Info Sheet/stock1-2.xlsx','./Info Sheet/stock2-3.xlsx']
    formator=StockData(paths)
    formator.pipline()
    formator=GuaranteeData('./Info Sheet/担保信息.xlsx')
    formator.pipline()
    formator=TransactionData('./Info Sheet/关联交易.xlsx')
    formator.pipline()




