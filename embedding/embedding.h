//
//  embedding.h
//  Pytorch
//
//  Created by 潘洪岩 on 2019/11/4.
//  Copyright © 2019 潘洪岩. All rights reserved.
//

#ifndef embedding_h
#define embedding_h

#include <torch/torch.h>

struct WordEmbedding:torch::nn::Module
{
    WordEmbedding(int vocab_size,int dim):vocab_size_(vocab_size),dim_(dim)
    {
        emb_=register_module("emb_",torch::nn::Embedding(vocab_size,dim));
        fc_=register_module("fc_", torch::nn::Linear(dim,vocab_size));
    }
    torch::Tensor forward(torch::Tensor& input)
    {
        torch::Tensor embeddings= emb_->forward(input);
        torch::Tensor embeddings_sum = torch::sum(embeddings,1);
        torch::Tensor output= fc_->forward(embeddings_sum);
        return torch::log_softmax(output,1);
    }
    int vocab_size_;
    int dim_;
    torch::nn::Embedding emb_{nullptr};
    torch::nn::Linear fc_{nullptr};
};

#endif /* embedding_h */
