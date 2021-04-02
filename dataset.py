from sys import base_prefix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils import open_file

class NMTDataset(Dataset):
    def __init__(self, src_path, tgt_path=None) :
        super(NMTDataset, self).__init__()
        self.src = open_file(src_path)
        self.tgt = None
        if tgt_path is not None:
            self.tgt = open_file(tgt_path)
        self.size = len(self.src)

    def __getitem__(self, index: int):
        if self.tgt is None:
            return self.src[index]
        return self.src[index], self.tgt[index]

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def save_file(data, file_path):
        with open(file_path, 'w', encoding="utf8") as f:
            for sent in data:
                f.write(sent+"\n")

        print(f"File written successfully at {file_path}")

    @staticmethod
    def create_train_test_valid_data(src_path, tgt_path):
        base_path = "./dataset/"
        x = NMTDataset.open_file(src_path)
        y = NMTDataset.open_file(tgt_path)
        src_train, src, tgt_train, tgt = train_test_split(x, y, test_size=0.2, shuffle=True)
        src_valid, src_test, tgt_valid, tgt_test = train_test_split(src, tgt, test_size=0.5, shuffle=True)
        NMTDataset.save_file(src_train, base_path+"src_train.txt")
        NMTDataset.save_file(tgt_train, base_path+"tgt_train.txt")
        NMTDataset.save_file(src_valid, base_path+"src_valid.txt")
        NMTDataset.save_file(tgt_valid, base_path+"tgt_valid.txt")
        NMTDataset.save_file(src_test, base_path+"src_test.txt")
        NMTDataset.save_file(tgt_test, base_path+"tgt_test.txt")

if __name__ == "__main__":
    NMTDataset.create_train_test_valid_data("./dataset/Newa_SS.txt", "./dataset/Eng_SS.txt")