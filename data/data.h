//
//  WordEmbedding.h
//  Pytorch
//
//  Created by 潘洪岩 on 2019/11/7.
//  Copyright © 2019 潘洪岩. All rights reserved.
//

#ifndef WordEmbedding_h
#define WordEmbedding_h

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>
#include "../vocab/vocab.hpp"

namespace torch {
namespace data{
namespace datasets{

struct Options{
    int ws=5;
    int dim=100;
    size_t train_batch_size=64;
    torch::DeviceType device = torch::kCPU;
    size_t log_interval = 1000;
    size_t epoch=20;
};

using Data = std::vector<std::string>;
struct Corpus{
    Vocab vocab;
    Data data;
};
class EmbeddingTextData:public torch::data::datasets::Dataset<EmbeddingTextData>
{
    using Example = torch::data::Example<>;
private:
    Data data_;
    Vocab vocab_;
    Options options_;
public:
    EmbeddingTextData(const Data& data,const Vocab& vocab, const Options& opts) : data_(data), vocab_(vocab),options_(opts) {}
    
    Example get(size_t index) {//生成句子的Tensor
        index+=options_.ws;//滑动出指定window大小的上文
        std::string word=data_[index];
        int idx=vocab_.get_id(word);
        auto target = torch::full(1,idx,torch::dtype(torch::kLong));
        auto context=torch::zeros(options_.ws,torch::dtype(torch::kLong));
        for(int i=options_.ws;i>=1;i--){
            size_t pos=index-i;
            word=data_[pos];
            int c_idx=vocab_.get_id(word);
            context[i-1]=c_idx;
        }
        return {context,target};
    }
    
    torch::optional<size_t> size() const {
        return data_.size()-options_.ws;
    }
};

/**
 *加载语料词典
 */
Corpus loadTrain(std::string& train_file)
{
    Corpus corpus;
    std::ifstream ifs(train_file);
    long long train_words=0;
    if (ifs.is_open()) {
        std::cout<<"Loading..."<<std::endl;
        std::string str;
        while (ifs >> str)
        {
            corpus.vocab.pushw(str);
            corpus.data.push_back(str);
            printf("%lldM%c", train_words / 1000000, 13);
            train_words++;
        }
        std::cout<<"TrainFile size:"<<corpus.vocab.tsize()<<"\nVocab size:"<<corpus.vocab.size()<<std::endl;
        ifs.close();
    }
    corpus.vocab.sort();
    return corpus;
}

}
}
}


#endif /* WordEmbedding_h */
