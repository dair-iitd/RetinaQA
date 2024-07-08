import json
import copy


class JsonLoader:
    def __init__(self, file_path, split, dataset, q_type="AU", is_inf=False):
        with open(file_path,  'r', encoding='UTF-8') as file:
            f = json.load(file)

            print(f"Preparing data for {split} : {len(f)}")

            if dataset == "webqsp":
                self.data = []
                for d in f:
                    if d["sketch"] == "null" and split == "train":
                        continue
                    self.data.append(d)

            elif dataset == "grailqa":
                self.data = f[:]

            elif dataset == "grailqability":
                if split == "train":
                    self.data = []

                    for d in f[:]:
                        d_copy = copy.deepcopy(d)

                        # take only answerable questions for A training (grailQAbility)
                        if ("qType" in d) and (q_type == "A") and (d["qType"] != "U"):
                            self.data.append(d_copy)

                        # for grailQA
                        elif not ("qType" in d) and (q_type == "A"):
                            self.data.append(d_copy)

                        # take non-NK questions for AU training
                        elif (q_type == "AU" ) and (d["s_expression"] != "NK"):
                            self.data.append(d_copy)


                elif split == "dev":
                    self.data = []
                    for d in f:
                        d_copy = copy.deepcopy(d)
                        if is_inf:
                            self.data.append(d_copy)
                        else:
                            if q_type == "A" and d["qType"] == "U":
                                pass
                            elif q_type == "AU" and d["s_expression"] == "NK":
                                pass
                            else:
                                self.data.append(d_copy)

                elif split == "train_nk":
                    self.data = []
                    for d in f:
                        if d["s_expression"] == "NK":
                            d_copy = copy.deepcopy(d)
                        else:
                            continue
                        self.data.append(d_copy)
                else:
                    self.data = f[:]
            
            self.len = len(self.data)
            print("length of data : ",self.len)
            self.file_path = file_path

        self.split = split
        self.question_id_to_idx_dict = dict()  # question_id: str -> idx: int
        self.build_question_id_to_idx_dict()

    def get_dataset_split(self):
        return self.split
        
    def build_question_id_to_idx_dict(self):
        for idx in range(0, self.len):
            question_id = self.get_question_id_by_idx(idx)
            self.question_id_to_idx_dict[str(question_id)] = idx

    def get_idx_by_question_id(self, question_id):
        question_id = str(question_id)
        if self.question_id_to_idx_dict is None or len(self.question_id_to_idx_dict) == 0:
            self.build_question_id_to_idx_dict()
        return self.question_id_to_idx_dict.get(question_id, -1)

    def get_sketch_by_idx(self, idx):
        return self.data[idx]['sketch']

    def get_question_id_by_idx(self, idx, format='int'):
        qid = self.data[idx]['qid']
        if format == 'str':
            return str(qid)
        return qid

    def get_question_by_idx(self, idx):
        return self.data[idx]['question']

    def get_len(self):
        return self.len

 