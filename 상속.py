# dictionary로 써보기

class my_TensorDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__()
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
    def __getitem__(self,index):
        return {'input_ids':self.tensors[0][index],'attention_mask':self.tensors[1][index],'token_type_ids':self.tensors[2][index],'labels':self.tensors[3][index],'labels_mask':self.tensors[4][index]}
      
    
