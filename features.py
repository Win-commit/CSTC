import pandas as pd
import os
'''
用于构建每种图的节点特征
'''
def create_metaFile():
    '''
    生成元数据文件
    '''
    metadata=pd.DataFrame(columns=['name', 'stock', 'guarantee','transaction','label'])
    df=pd.read_excel('./Info Sheet/股东信息.xlsx', index_col=0)
    values=df['企业名称'].unique()
    df1=pd.read_csv('./Stock/mapped.csv', index_col=0)
    df2=pd.read_csv('./Guarantee/mapped.csv', index_col=0)
    df3=pd.read_csv('./Transaction/mapped.csv', index_col=0)
    df4=pd.read_excel('./Info Sheet/labels.xlsx', index_col=0)
    for value in values:
        metadata=metadata._append({
            'name':value, 
            'stock':df1[df1['Fullname']==value]['ID'].item(), 
            'guarantee':df2[df2['Fullname']==value]['ID'].item(), 
            'transaction':df3[df3['Fullname']==value]['ID'].item(), 
            'label':df4[df4['公司名称']==value]['label_encoded'].item()},
            ignore_index=True)
    if not os.path.exists('Meta'):
            os.makedirs('Meta')
    metadata.to_csv('./Meta/metadata.csv')



def Stock_features(info_paths,meta_path):
    '''
    Info paths include shareholders info and stock0-1 info paths
    '''
    features=pd.DataFrame(columns=['ID', '风险集中', '股权关联度','股权集中度'])
    metadata=pd.read_csv(meta_path, index_col=0)
    metadata=metadata[['stock','name']]
    shareInfo=pd.read_excel(info_paths[0], index_col=0)
    stockInfo=pd.read_excel(info_paths[1], index_col=0)
    map1={}
    map2={}
    map3={}
    for index, row in metadata.iterrows():
        ID=row['stock']
        share=shareInfo[shareInfo['企业名称']==row['name']]
        col_ls=[f'大股东持股比例第{i}名' for i in range(1,11)]
        result1 = (share[col_ls[0:5]] > 10).all(axis=1).item()
        result3= share[col_ls[0]].item()
        shareholders = share[col_ls].notnull().sum().sum()
        counts = stockInfo.groupby('企业名称')['控股企业名称'].count()
        result2 = counts.loc[row['name']].item() + shareholders
        map1[ID]=result1
        map2[ID]=result2
        map3[ID]=result3
    for key in map1.keys():
        features=features._append({'ID':key, 
                                   '风险集中':map1[key], 
                                   '股权关联度':map2[key], 
                                   '股权集中度':map3[key]}, 
                                   ignore_index=True)
    features.to_csv('./Stock/features.csv')


def Transaction_features(transaction_path,meta_path):
    features=pd.DataFrame(columns=['ID', '大额交易', '交易广度'])
    metadata=pd.read_csv(meta_path, index_col=0)
    metadata=metadata[['transaction','name']]

    transactionInfo=pd.read_excel(transaction_path, index_col=0)
    transactionInfo['交易金额(万元)']=transactionInfo['交易金额(万元)'].fillna(0)
    map1={}
    map2={}
    for index, row in metadata.iterrows():
        ID=row['transaction']
        
        transaction=transactionInfo.copy()
        count1 = transaction.groupby('公司名称')['交易金额(万元)'].apply(lambda x: (x > 200).sum())
        count2 = transaction.groupby('公司名称')['关联方'].count()
        result1 = count1.loc[row['name']].item()
        result2 = count2.loc[row['name']].item()
        map1[ID]=result1/result2
        map2[ID]=result2
    for key in map1.keys():
        features=features._append({'ID':key, 
                                   '大额交易':map1[key], 
                                   '交易广度':map2[key]}, 
                                   ignore_index=True)
    features.to_csv('./Transaction/features.csv')



def Guarantee_features(guarantee_path,meta_path,transaction_path):
    features=pd.DataFrame(columns=['ID', '担保总额', '担保关联度','担保交易'])
    metadata=pd.read_csv(meta_path, index_col=0)
    metadata=metadata[['guarantee','name']]
    map1={}
    map2={}
    map3={}
    guaranteeInfo=pd.read_excel(guarantee_path, index_col=0)
    guaranteeInfo['担保金额(万元)']=guaranteeInfo['担保金额(万元)'].fillna(0)
    transactionInfo=pd.read_excel(transaction_path, index_col=0)
    for index, row in metadata.iterrows():
        ID=row['guarantee']
        guarantee=guaranteeInfo.copy()
        amount=guarantee.groupby('担保方公司名称')['担保金额(万元)'].sum()
        nums=guarantee.groupby('担保方公司名称')['被担保方公司名称'].count()
        result1 = amount.loc[row['name']].item()
        result2= nums.loc[row['name']].item()
        map1[ID]=result1
        map2[ID]=result2
        guarted=guarantee[guarantee['担保方公司名称']==row['name']]['被担保方公司名称'].unique()
        result = transactionInfo.groupby('公司名称')['关联方'].apply(lambda x: x.isin(guarted).any())
        result3 = result.loc[row['name']].item()
        map3[ID]=result3
    for key in map1.keys():
        features=features._append({'ID':key, 
                                   '担保总额':map1[key], 
                                   '担保关联度':map2[key], 
                                   '担保交易':map3[key]}, 
                                   ignore_index=True)
    features.to_csv('./Guarantee/features.csv')



    


if __name__ == "__main__":
    create_metaFile()
    # metaPath='./Meta/metadata.csv'
    # stockPath=['./Info Sheet/股东信息.xlsx','./Info Sheet/stock0-1.xlsx']
    # guaranteePath='./Info Sheet/担保信息.xlsx'
    # transactionPath='./Info Sheet/关联交易.xlsx'
    # Stock_features(stockPath,metaPath)
    # Transaction_features(transactionPath,metaPath)
    # Guarantee_features(guaranteePath,metaPath,transactionPath)