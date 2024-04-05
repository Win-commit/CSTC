from G_Database import CompanyDataset
from tqdm import tqdm
from dgl import batch as dgl_batch
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import Finsentry
import dgl
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def collate_fn(samples):
    # samples是一个列表，每个元素对应CompanyDataset.__getitem__的输出
    # 即，每个元素是([stock_graph, guarantee_graph, transaction_graph], [label1, label2])
    
    # 分别处理图和标签
    batched_graphs = [[], [], []]  # 用于存储每种类型图的批处理列表
    labels = []  # 用于存储标签
    
    for sample in samples:
        graphs, label = sample
        for i in range(3):  # 遍历每个样本的三张图
            batched_graphs[i].append(graphs[i])
        labels.append(label)  # 收集标签
    
    # 对每种类型的图进行批处理
    for i in range(3):
        batched_graphs[i] = dgl_batch(batched_graphs[i])
    
    # 将标签转换为Tensor
    labels = torch.tensor(labels, dtype=torch.float32)

    # 注意，返回的是一个元组，其中包含三个批处理的图
    return batched_graphs, labels




def train(model: torch.nn.Module, optimizer, trainloader, device):
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = [g.to(device) for g in batch_graphs]
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs)
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches

@torch.no_grad()
def test(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = [g.to(device) for g in batch_graphs]
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs)
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()
    return correct / num_graphs, loss / num_graphs


def main(args):
    #prepare data--------------------------------
    meta_path='./Meta/metadata.csv'
    dataset = CompanyDataset(meta_path)


    dataset_size = len(dataset)
    num_training = int(len(dataset) * 0.8)
    num_test = dataset_size - num_training
    train_set, test_set = torch.utils.data.random_split(dataset, [num_training, num_test])
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn, shuffle=False)

    #prepare model-------------------------------
    model = Finsentry(in_dim=[3,3,2], 
                      hidden_dim=[16,16,8], 
                      out_dim=32, 
                      n_layers=3).to(device)
    

    #prepare training components----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #training--------------------------------
    for epoch in tqdm(range(100)):
        loss = train(model, optimizer, train_loader, device)
        acc, test_loss = test(model, test_loader, device)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {acc:.4f}")
    
    #save model-------------------------------
    torch.save(model.state_dict(), 'Finsentry.pth')

if __name__ == '__main__':
    main(None)
